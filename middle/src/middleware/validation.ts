import type { Request, Response, NextFunction } from 'express'

interface ValidationRule {
  field: string
  type: 'string' | 'number' | 'boolean' | 'object' | 'array'
  required?: boolean
  min?: number
  max?: number
  pattern?: RegExp
  message?: string
}

export function validateBody(rules: ValidationRule[]) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const errors: string[] = []

    for (const rule of rules) {
      const value = req.body?.[rule.field]

      if (rule.required && (value === undefined || value === null || value === '')) {
        errors.push(rule.message || `${rule.field} is required`)
        continue
      }

      if (value === undefined || value === null) continue

      if (rule.type === 'string' && typeof value !== 'string') {
        errors.push(`${rule.field} must be a string`)
      } else if (rule.type === 'number' && typeof value !== 'number') {
        errors.push(`${rule.field} must be a number`)
      } else if (rule.type === 'boolean' && typeof value !== 'boolean') {
        errors.push(`${rule.field} must be a boolean`)
      } else if (rule.type === 'object' && (typeof value !== 'object' || Array.isArray(value))) {
        errors.push(`${rule.field} must be an object`)
      } else if (rule.type === 'array' && !Array.isArray(value)) {
        errors.push(`${rule.field} must be an array`)
      }

      if (rule.type === 'string' && typeof value === 'string') {
        if (rule.min !== undefined && value.length < rule.min) {
          errors.push(`${rule.field} must be at least ${rule.min} characters`)
        }
        if (rule.max !== undefined && value.length > rule.max) {
          errors.push(`${rule.field} must be at most ${rule.max} characters`)
        }
        if (rule.pattern && !rule.pattern.test(value)) {
          errors.push(rule.message || `${rule.field} has invalid format`)
        }
      }

      if (rule.type === 'number' && typeof value === 'number') {
        if (rule.min !== undefined && value < rule.min) {
          errors.push(`${rule.field} must be at least ${rule.min}`)
        }
        if (rule.max !== undefined && value > rule.max) {
          errors.push(`${rule.field} must be at most ${rule.max}`)
        }
      }
    }

    if (errors.length > 0) {
      res.status(400).json({ error: 'Validation failed', details: errors })
      return
    }

    next()
  }
}

export function validateQuery(rules: ValidationRule[]) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const errors: string[] = []

    for (const rule of rules) {
      const value = req.query[rule.field]

      if (rule.required && (value === undefined || value === '')) {
        errors.push(rule.message || `${rule.field} is required`)
        continue
      }

      if (value === undefined) continue

      if (rule.type === 'number') {
        const num = Number(value)
        if (isNaN(num)) {
          errors.push(`${rule.field} must be a number`)
        } else if (rule.min !== undefined && num < rule.min) {
          errors.push(`${rule.field} must be at least ${rule.min}`)
        } else if (rule.max !== undefined && num > rule.max) {
          errors.push(`${rule.field} must be at most ${rule.max}`)
        }
      }
    }

    if (errors.length > 0) {
      res.status(400).json({ error: 'Validation failed', details: errors })
      return
    }

    next()
  }
}
