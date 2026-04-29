export function isValidUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

export function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
}

export function isValidApiKey(key: string): boolean {
  return /^(ak-|sk-)?[a-zA-Z0-9]{16,64}$/.test(key)
}

export function isValidCron(expr: string): boolean {
  const parts = expr.trim().split(/\s+/)
  if (parts.length !== 5) return false
  for (const part of parts) {
    if (!/^(\*|\d+(-\d+)?(,\d+(-\d+)?)*|\d+\/\d+)$/.test(part) && part !== '*') return false
  }
  return true
}

export function isValidPort(port: number): boolean {
  return Number.isInteger(port) && port > 0 && port <= 65535
}

export function isAscii(text: string): boolean {
  return Array.from(text).every((char) => char.charCodeAt(0) <= 0x7f)
}

export function isJsonString(text: string): boolean {
  try {
    JSON.parse(text)
    return true
  } catch {
    return false
  }
}

export function validateRequired(value: unknown, fieldName: string): string | null {
  if (value === undefined || value === null || value === '') return `${fieldName} is required`
  return null
}

export function validateMinLength(value: string, min: number, fieldName: string): string | null {
  if (value.length < min) return `${fieldName} must be at least ${min} characters`
  return null
}

export function validateMaxLength(value: string, max: number, fieldName: string): string | null {
  if (value.length > max) return `${fieldName} must be at most ${max} characters`
  return null
}

export function validateRange(value: number, min: number, max: number, fieldName: string): string | null {
  if (value < min || value > max) return `${fieldName} must be between ${min} and ${max}`
  return null
}
