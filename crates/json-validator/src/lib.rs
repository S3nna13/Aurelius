use napi_derive::napi;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[napi(object)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub field_count: u32,
    pub depth: u32,
}

#[napi(object)]
pub struct TypeCheckResult {
    pub is_valid: bool,
    pub expected_type: String,
    pub actual_type: String,
    pub message: String,
}

#[napi(object)]
pub struct JsonStats {
    pub size_bytes: u32,
    pub field_count: u32,
    pub depth: u32,
    pub max_array_length: u32,
    pub types_present: Vec<String>,
    pub has_nulls: bool,
    pub has_nested_objects: bool,
}

fn get_depth(val: &Value) -> u32 {
    let mut max_depth = 1u32;
    let mut stack = vec![(val, 1u32)];
    while let Some((current, depth)) = stack.pop() {
        if depth > max_depth {
            max_depth = depth;
        }
        if max_depth > 1000 {
            break;
        }
        match current {
            Value::Object(map) => {
                for child in map.values() {
                    stack.push((child, depth + 1));
                }
            }
            Value::Array(arr) => {
                for child in arr {
                    stack.push((child, depth + 1));
                }
            }
            _ => {}
        }
    }
    max_depth
}

fn count_fields(val: &Value) -> u32 {
    let mut count = 0u32;
    let mut stack = vec![val];
    while let Some(current) = stack.pop() {
        match current {
            Value::Object(map) => {
                count = count.saturating_add(map.len() as u32);
                for child in map.values() {
                    stack.push(child);
                }
            }
            Value::Array(arr) => {
                count = count.saturating_add(arr.len() as u32);
                for child in arr {
                    stack.push(child);
                }
            }
            _ => {}
        }
    }
    count
}

fn get_type_name(val: &Value) -> String {
    match val {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "boolean".to_string(),
        Value::Number(n) if n.is_f64() => "number".to_string(),
        Value::Number(_) => "integer".to_string(),
        Value::String(_) => "string".to_string(),
        Value::Array(_) => "array".to_string(),
        Value::Object(_) => "object".to_string(),
    }
}

#[napi]
pub struct JsonValidator {
    max_depth: u32,
    max_field_count: u32,
    max_array_length: u32,
}

#[napi]
impl JsonValidator {
    #[napi(constructor)]
    pub fn new() -> Self {
        JsonValidator {
            max_depth: 50,
            max_field_count: 10000,
            max_array_length: 100000,
        }
    }

    #[napi]
    pub fn with_limits(max_depth: u32, max_field_count: u32, max_array_length: u32) -> Self {
        JsonValidator {
            max_depth,
            max_field_count,
            max_array_length,
        }
    }

    #[napi]
    pub fn validate(&self, json_str: String) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        let value: Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return ValidationResult {
                    valid: false,
                    errors: vec![format!("Invalid JSON: {}", e)],
                    warnings: vec![],
                    field_count: 0,
                    depth: 0,
                };
            }
        };

        let depth = get_depth(&value);
        let field_count = count_fields(&value);

        if depth > self.max_depth {
            errors.push(format!(
                "Max depth exceeded: {} > {}",
                depth, self.max_depth
            ));
        }

        if field_count > self.max_field_count {
            errors.push(format!(
                "Max field count exceeded: {} > {}",
                field_count, self.max_field_count
            ));
        }

        if let Value::Array(arr) = &value {
            if arr.len() as u32 > self.max_array_length {
                warnings.push(format!(
                    "Large array: {} elements (limit: {})",
                    arr.len(),
                    self.max_array_length
                ));
            }
        }

        // Check for common issues
        if value.is_null() {
            warnings.push("Root value is null".to_string());
        }

        if field_count == 0 && !value.is_null() && !value.is_string() && !value.is_number() {
            warnings.push("Empty object or array".to_string());
        }

        self.validate_value(&value, &mut errors, &mut warnings, "#", 0);

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            field_count,
            depth,
        }
    }

    fn validate_value(
        &self,
        val: &Value,
        errors: &mut Vec<String>,
        warnings: &mut Vec<String>,
        path: &str,
        current_depth: u32,
    ) {
        if current_depth > self.max_depth {
            return;
        }

        match val {
            Value::Object(map) => {
                if map.is_empty() {
                    warnings.push(format!("Empty object at {}", path));
                }
                for (key, child) in map {
                    let child_path = format!("{}.{}", path, key);
                    self.validate_value(child, errors, warnings, &child_path, current_depth + 1);
                }
            }
            Value::Array(arr) => {
                for (i, child) in arr.iter().enumerate() {
                    let child_path = format!("{}[{}]", path, i);
                    self.validate_value(child, errors, warnings, &child_path, current_depth + 1);
                }
            }
            _ => {}
        }
    }

    #[napi]
    pub fn validate_schema(&self, json_str: String, schema: SchemaDefinition) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        let value: Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return ValidationResult {
                    valid: false,
                    errors: vec![format!("Invalid JSON: {}", e)],
                    warnings: vec![],
                    field_count: 0,
                    depth: 0,
                };
            }
        };

        let schema = match schema.into_internal() {
            Ok(schema) => schema,
            Err(err) => {
                return ValidationResult {
                    valid: false,
                    errors: vec![format!("Invalid schema: {}", err)],
                    warnings: vec![],
                    field_count: count_fields(&value),
                    depth: get_depth(&value),
                };
            }
        };

        self.validate_against_schema(&value, &schema, &mut errors, &mut warnings, "#");

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            field_count: count_fields(&value),
            depth: get_depth(&value),
        }
    }

    fn validate_against_schema(
        &self,
        value: &Value,
        schema: &SchemaDefinitionNode,
        errors: &mut Vec<String>,
        warnings: &mut Vec<String>,
        path: &str,
    ) {
        let actual_type = get_type_name(value);

        // Check null
        if value.is_null() && !schema.nullable.unwrap_or(false) {
            errors.push(format!("{}: expected {}, got null", path, schema.type_name));
            return;
        }
        if value.is_null() {
            return;
        }

        // Check type
        if schema.type_name != "any" && actual_type != schema.type_name {
            errors.push(format!(
                "{}: type mismatch — expected {}, got {}",
                path, schema.type_name, actual_type
            ));
            return;
        }

        // Object validation
        if let Value::Object(map) = value {
            if let Some(required) = &schema.required_fields {
                for field in required {
                    if !map.contains_key(field) {
                        errors.push(format!("{}: missing required field '{}'", path, field));
                    }
                }
            }

            if let Some(min) = schema.min_fields {
                if (map.len() as u32) < min {
                    errors.push(format!(
                        "{}: minimum {} fields, got {}",
                        path,
                        min,
                        map.len()
                    ));
                }
            }
            if let Some(max) = schema.max_fields {
                if (map.len() as u32) > max {
                    errors.push(format!(
                        "{}: maximum {} fields, got {}",
                        path,
                        max,
                        map.len()
                    ));
                }
            }

            for (key, child) in map {
                let child_path = format!("{}.{}", path, key);
                // Check nested schemas
                if let Some(nested) = &schema.properties {
                    if let Some(field_schema) = nested.iter().find(|prop| prop.name == *key) {
                        self.validate_against_schema(
                            child,
                            &field_schema.schema,
                            errors,
                            warnings,
                            &child_path,
                        );
                    }
                }
            }
        }

        // Array validation
        if let Value::Array(arr) = value {
            if let Some(min) = schema.min_items {
                if (arr.len() as u32) < min {
                    errors.push(format!(
                        "{}: minimum {} items, got {}",
                        path,
                        min,
                        arr.len()
                    ));
                }
            }
            if let Some(max) = schema.max_items {
                if (arr.len() as u32) > max {
                    errors.push(format!(
                        "{}: maximum {} items, got {}",
                        path,
                        max,
                        arr.len()
                    ));
                }
            }
            if let Some(item_schema) = &schema.items_schema {
                for (i, item) in arr.iter().enumerate() {
                    let child_path = format!("{}[{}]", path, i);
                    self.validate_against_schema(item, item_schema, errors, warnings, &child_path);
                }
            }
        }

        // String validation
        if let Value::String(s) = value {
            if let Some(min) = schema.min_length {
                if (s.len() as u32) < min {
                    errors.push(format!("{}: minimum {} chars, got {}", path, min, s.len()));
                }
            }
            if let Some(max) = schema.max_length {
                if (s.len() as u32) > max {
                    errors.push(format!("{}: maximum {} chars, got {}", path, max, s.len()));
                }
            }
            if let Some(pattern) = &schema.pattern {
                match regex::Regex::new(pattern) {
                    Ok(re) => {
                        if !re.is_match(s) {
                            errors.push(format!("{}: does not match pattern '{}'", path, pattern));
                        }
                    }
                    Err(e) => {
                        errors.push(format!(
                            "{}: invalid regex pattern '{}': {}",
                            path, pattern, e
                        ));
                    }
                }
            }
        }

        // Numeric validation
        if let Value::Number(n) = value {
            let f = n.as_f64().unwrap_or(0.0);
            if let Some(min) = schema.minimum {
                if f < min {
                    errors.push(format!("{}: minimum {}, got {}", path, min, f));
                }
            }
            if let Some(max) = schema.maximum {
                if f > max {
                    errors.push(format!("{}: maximum {}, got {}", path, max, f));
                }
            }
        }
    }

    #[napi]
    pub fn analyze_json(&self, json_str: String) -> JsonStats {
        let value: Value = serde_json::from_str(&json_str).unwrap_or(Value::Null);

        let mut types = std::collections::HashSet::new();
        let mut max_arr_len = 0u32;
        let mut has_nulls = false;
        let mut has_nested = false;

        self.analyze_value(
            &value,
            &mut types,
            &mut max_arr_len,
            &mut has_nulls,
            &mut has_nested,
            0,
        );

        JsonStats {
            size_bytes: u32::try_from(json_str.len()).unwrap_or(u32::MAX),
            field_count: count_fields(&value),
            depth: get_depth(&value),
            max_array_length: max_arr_len,
            types_present: types.into_iter().collect(),
            has_nulls,
            has_nested_objects: has_nested,
        }
    }

    fn analyze_value(
        &self,
        val: &Value,
        types: &mut std::collections::HashSet<String>,
        max_arr: &mut u32,
        has_nulls: &mut bool,
        has_nested: &mut bool,
        depth: u32,
    ) {
        if depth > self.max_depth {
            return;
        }
        types.insert(get_type_name(val));

        match val {
            Value::Null => *has_nulls = true,
            Value::Object(map) => {
                if depth > 0 {
                    *has_nested = true;
                }
                for child in map.values() {
                    self.analyze_value(child, types, max_arr, has_nulls, has_nested, depth + 1);
                }
            }
            Value::Array(arr) => {
                *max_arr = (*max_arr).max(arr.len() as u32);
                for child in arr {
                    self.analyze_value(child, types, max_arr, has_nulls, has_nested, depth + 1);
                }
            }
            _ => {}
        }
    }

    #[napi]
    pub fn type_check(&self, json_str: String, expected_type: String) -> TypeCheckResult {
        let value: Value = serde_json::from_str(&json_str).unwrap_or(Value::Null);
        let actual = get_type_name(&value);
        let is_valid = expected_type == "any" || actual == expected_type;
        let message = if is_valid {
            "Type check passed".to_string()
        } else {
            format!("Expected {}, got {}", expected_type, actual)
        };

        TypeCheckResult {
            is_valid,
            expected_type,
            actual_type: actual,
            message,
        }
    }
}

#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaDefinition {
    pub type_name: String,
    pub nullable: Option<bool>,
    pub required_fields: Option<Vec<String>>,
    pub properties: Option<Vec<SchemaProperty>>,
    pub items_schema: Option<Value>,
    pub min_fields: Option<u32>,
    pub max_fields: Option<u32>,
    pub min_items: Option<u32>,
    pub max_items: Option<u32>,
    pub min_length: Option<u32>,
    pub max_length: Option<u32>,
    pub pattern: Option<String>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaProperty {
    pub name: String,
    pub schema: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SchemaDefinitionNode {
    type_name: String,
    nullable: Option<bool>,
    required_fields: Option<Vec<String>>,
    properties: Option<Vec<SchemaPropertyNode>>,
    items_schema: Option<Box<SchemaDefinitionNode>>,
    min_fields: Option<u32>,
    max_fields: Option<u32>,
    min_items: Option<u32>,
    max_items: Option<u32>,
    min_length: Option<u32>,
    max_length: Option<u32>,
    pattern: Option<String>,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SchemaPropertyNode {
    name: String,
    schema: SchemaDefinitionNode,
}

impl SchemaDefinition {
    fn into_internal(self) -> Result<SchemaDefinitionNode, String> {
        let value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        serde_json::from_value(value).map_err(|e| e.to_string())
    }
}
