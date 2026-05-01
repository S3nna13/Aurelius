export function uuidV4(): string
export function uuidV7(): string
export function uuidV4Batch(count: number): string[]
export function uuidV7Batch(count: number): string[]
export function uuidNil(): string
export function uuidIsValid(s: string): boolean
export function uuidTimestamp(uuidStr: string): number | null
export function uuidVersion(uuidStr: string): number | null
export function uuidVariant(uuidStr: string): string
export function uuidShort(): string
export function uuidSortKey(): number
