export interface ValidationResult { valid: boolean; errors: string[]; warnings: string[]; fieldCount: number; depth: number }
export interface TypeCheckResult { isValid: boolean; expectedType: string; actualType: string; message: string }
export interface JsonStats { sizeBytes: number; fieldCount: number; depth: number; maxArrayLength: number; typesPresent: string[]; hasNulls: boolean; hasNestedObjects: boolean }
export interface SchemaDefinition { typeName: string; nullable?: boolean; requiredFields?: string[]; properties?: SchemaProperty[]; itemsSchema?: SchemaDefinition; minFields?: number; maxFields?: number; minItems?: number; maxItems?: number; minLength?: number; maxLength?: number; pattern?: string; minimum?: number; maximum?: number }
export interface SchemaProperty { name: string; schema: SchemaDefinition }

export declare class JsonValidator {
  constructor()
  static withLimits(maxDepth: number, maxFieldCount: number, maxArrayLength: number): JsonValidator
  validate(jsonStr: string): ValidationResult
  validateSchema(jsonStr: string, schema: SchemaDefinition): ValidationResult
  analyzeJson(jsonStr: string): JsonStats
  typeCheck(jsonStr: string, expectedType: string): TypeCheckResult
}
