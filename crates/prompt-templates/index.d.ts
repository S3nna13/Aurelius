export interface RenderedPrompt { text: string; variablesUsed: string[]; variablesMissing: string[]; tokenEstimate: number }
export interface TemplateInfo { name: string; variableCount: number; variables: string[]; hasConditionals: boolean; hasLoops: boolean; characterCount: number }
export interface ChatMLMessage { role: string; content: string }

export declare class PromptEngine {
  constructor()
  registerTemplate(name: string, template: string): void
  registerPartial(name: string, content: string): void
  getTemplate(name: string): string | null
  getPartial(name: string): string | null
  listTemplates(): string[]
  listPartials(): string[]
  deleteTemplate(name: string): boolean
  deletePartial(name: string): boolean
  render(templateName: string, variables: Record<string, string>): RenderedPrompt
  renderString(template: string, variables: Record<string, string>): RenderedPrompt
  analyze(templateName: string): TemplateInfo
  extractVariables(template: string): string[]
  validateTemplate(template: string, requiredVars: string[]): string[]
  renderChatML(system: string | null, messages: ChatMLMessage[], variables: Record<string, string>): RenderedPrompt
  renderLlama3(system: string | null, messages: ChatMLMessage[], variables: Record<string, string>): RenderedPrompt
  estimateTokensBatch(texts: string[]): number[]
}
