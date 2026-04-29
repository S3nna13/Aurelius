use std::collections::HashMap;

use napi::Error;
use napi_derive::napi;
use regex::Regex;

#[napi(object)]
pub struct RenderedPrompt {
    pub text: String,
    pub variables_used: Vec<String>,
    pub variables_missing: Vec<String>,
    pub token_estimate: u32,
}

#[napi(object)]
pub struct TemplateInfo {
    pub name: String,
    pub variable_count: u32,
    pub variables: Vec<String>,
    pub has_conditionals: bool,
    pub has_loops: bool,
    pub character_count: u32,
}

#[napi(object)]
pub struct PromptPartial {
    pub name: String,
    pub content: String,
    pub required_variables: Vec<String>,
}

fn extract_variables(template: &str) -> Vec<String> {
    let re = Regex::new(r"\{\{(\w+)\}\}").unwrap();
    let mut vars: Vec<String> = re
        .captures_iter(template)
        .map(|c| c[1].to_string())
        .collect();
    vars.sort();
    vars.dedup();
    vars
}

fn estimate_tokens(text: &str) -> u32 {
    (text.split_whitespace().count() as f64 * 1.3) as u32
}

#[napi]
pub struct PromptEngine {
    templates: HashMap<String, String>,
    partials: HashMap<String, String>,
}

#[napi]
impl PromptEngine {
    #[napi(constructor)]
    pub fn new() -> Self {
        PromptEngine {
            templates: HashMap::new(),
            partials: HashMap::new(),
        }
    }

    #[napi]
    pub fn register_template(&mut self, name: String, template: String) {
        self.templates.insert(name, template);
    }

    #[napi]
    pub fn register_partial(&mut self, name: String, content: String) {
        self.partials.insert(name, content);
    }

    #[napi]
    pub fn get_template(&self, name: String) -> Option<String> {
        self.templates.get(&name).cloned()
    }

    #[napi]
    pub fn get_partial(&self, name: String) -> Option<String> {
        self.partials.get(&name).cloned()
    }

    #[napi]
    pub fn list_templates(&self) -> Vec<String> {
        let mut names: Vec<String> = self.templates.keys().cloned().collect();
        names.sort();
        names
    }

    #[napi]
    pub fn list_partials(&self) -> Vec<String> {
        let mut names: Vec<String> = self.partials.keys().cloned().collect();
        names.sort();
        names
    }

    #[napi]
    pub fn delete_template(&mut self, name: String) -> bool {
        self.templates.remove(&name).is_some()
    }

    #[napi]
    pub fn delete_partial(&mut self, name: String) -> bool {
        self.partials.remove(&name).is_some()
    }

    #[napi]
    pub fn render(
        &self,
        template_name: String,
        variables: HashMap<String, String>,
    ) -> napi::Result<RenderedPrompt> {
        let template = self
            .templates
            .get(&template_name)
            .ok_or_else(|| Error::from_reason(format!("Template '{}' not found", template_name)))?;

        self.render_string(template.clone(), variables)
    }

    #[napi]
    pub fn render_string(
        &self,
        template: String,
        variables: HashMap<String, String>,
    ) -> napi::Result<RenderedPrompt> {
        let mut used = Vec::new();
        let mut missing = Vec::new();

        // First pass: resolve partials: {{>partial_name}}
        let partial_re = Regex::new(r"\{\{>(\w+)\}\}").unwrap();
        let with_partials = partial_re
            .replace_all(&template, |caps: &regex::Captures| {
                let partial_name = &caps[1];
                self.partials
                    .get(partial_name)
                    .cloned()
                    .unwrap_or_else(|| format!("{{{{>{}?}}}}", partial_name))
            })
            .to_string();

        // Second pass: replace variables
        let var_re = Regex::new(r"\{\{(\w+)\}\}").unwrap();
        let rendered = var_re
            .replace_all(&with_partials, |caps: &regex::Captures| {
                let var_name = &caps[1];
                match variables.get(var_name) {
                    Some(val) => {
                        used.push(var_name.to_string());
                        val.clone()
                    }
                    None => {
                        missing.push(var_name.to_string());
                        format!("{{{{{}}}}}", var_name)
                    }
                }
            })
            .to_string();

        // Remove any remaining unresolved variables
        let final_text = var_re.replace_all(&rendered, "").to_string();

        Ok(RenderedPrompt {
            token_estimate: estimate_tokens(&final_text),
            text: final_text,
            variables_used: used,
            variables_missing: missing,
        })
    }

    #[napi]
    pub fn analyze(&self, template_name: String) -> napi::Result<TemplateInfo> {
        let template = self
            .templates
            .get(&template_name)
            .ok_or_else(|| Error::from_reason(format!("Template '{}' not found", template_name)))?;

        let variables = extract_variables(template);
        let has_partials = template.contains("{{>");
        let has_conditionals = template.contains("{{#")
            || template.contains("{{^")
            || template.contains("{{/");
        let has_loops = template.contains("{{#each")
            || template.contains("{{#for");

        Ok(TemplateInfo {
            name: template_name,
            variable_count: variables.len() as u32,
            variables,
            has_conditionals: has_conditionals || has_partials,
            has_loops,
            character_count: template.len() as u32,
        })
    }

    #[napi]
    pub fn extract_variables(&self, template: String) -> Vec<String> {
        extract_variables(&template)
    }

    #[napi]
    pub fn validate_template(
        &self,
        template: String,
        required_vars: Vec<String>,
    ) -> Vec<String> {
        let found = extract_variables(&template);
        let mut missing = Vec::new();
        for var in &required_vars {
            if !found.contains(var) {
                missing.push(var.clone());
            }
        }
        missing
    }

    #[napi]
    pub fn render_chatml(
        &self,
        system: Option<String>,
        messages: Vec<ChatMLMessage>,
        variables: HashMap<String, String>,
    ) -> RenderedPrompt {
        let mut parts = Vec::new();

        for msg in &messages {
            if let Some(err) = validate_role(&msg.role) {
                return RenderedPrompt {
                    text: format!("Error: {}", err),
                    variables_used: vec![],
                    variables_missing: vec![],
                    token_estimate: 0,
                };
            }
        }

        if let Some(sys) = system {
            let sanitized = sanitize_content(&sys);
            let rendered = self
                .render_string(sanitized, variables.clone())
                .unwrap_or(RenderedPrompt {
                    text: sys,
                    variables_used: vec![],
                    variables_missing: vec![],
                    token_estimate: 0,
                });
            parts.push(format!("<|im_start|>system\n{}<|im_end|>", rendered.text));
        }

        for msg in &messages {
            let sanitized = sanitize_content(&msg.content);
            let rendered = self
                .render_string(sanitized, variables.clone())
                .unwrap_or(RenderedPrompt {
                    text: msg.content.clone(),
                    variables_used: vec![],
                    variables_missing: vec![],
                    token_estimate: 0,
                });
            parts.push(format!(
                "<|im_start|>{}\n{}<|im_end|>",
                msg.role, rendered.text
            ));
        }

        parts.push("<|im_start|>assistant\n".to_string());

        let text = parts.join("\n");
        RenderedPrompt {
            token_estimate: estimate_tokens(&text),
            text,
            variables_used: vec![],
            variables_missing: vec![],
        }
    }

    #[napi]
    pub fn render_llama3(
        &self,
        system: Option<String>,
        messages: Vec<ChatMLMessage>,
        variables: HashMap<String, String>,
    ) -> RenderedPrompt {
        let mut parts = Vec::new();

        for msg in &messages {
            if let Some(e) = validate_role(&msg.role) {
                return RenderedPrompt {
                    text: format!("Error: {}", e),
                    variables_used: vec![],
                    variables_missing: vec![],
                    token_estimate: 0,
                };
            }
        }

        parts.push("<|begin_of_text|>".to_string());

        if let Some(sys) = system {
            let sanitized = sanitize_content(&sys);
            let rendered = self
                .render_string(sanitized, variables.clone())
                .unwrap_or(RenderedPrompt {
                    text: sys,
                    variables_used: vec![],
                    variables_missing: vec![],
                    token_estimate: 0,
                });
            parts.push(format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                rendered.text
            ));
        }

        for msg in &messages {
            let sanitized = sanitize_content(&msg.content);
            let rendered = self
                .render_string(sanitized, variables.clone())
                .unwrap_or(RenderedPrompt {
                    text: msg.content.clone(),
                    variables_used: vec![],
                    variables_missing: vec![],
                    token_estimate: 0,
                });
            parts.push(format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.role, rendered.text
            ));
        }

        parts.push("<|start_header_id|>assistant<|end_header_id|>\n\n".to_string());

        let text = parts.join("");
        RenderedPrompt {
            token_estimate: estimate_tokens(&text),
            text,
            variables_used: vec![],
            variables_missing: vec![],
        }
    }

    #[napi]
    pub fn estimate_tokens_batch(&self, texts: Vec<String>) -> Vec<u32> {
        texts.iter().map(|t| estimate_tokens(t)).collect()
    }
}

const VALID_ROLES: &[&str] = &["system", "user", "assistant", "tool"];

fn validate_role(role: &str) -> Option<String> {
    if !VALID_ROLES.contains(&role) {
        return Some(format!("Invalid role '{}': must be one of {:?}", role, VALID_ROLES));
    }
    if role.contains(char::is_whitespace) || role.contains('<') || role.contains('>') {
        return Some(format!("Invalid role '{}': contains forbidden characters", role));
    }
    None
}

fn sanitize_content(content: &str) -> String {
    content.replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .replace("<|eot_id|>", "")
        .replace("<|begin_of_text|>", "")
        .replace("<|end_of_text|>", "")
        .replace("<|start_header_id|>", "")
        .replace("<|end_header_id|>", "")
}

#[napi(object)]
pub struct ChatMLMessage {
    pub role: String,
    pub content: String,
}
