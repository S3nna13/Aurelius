import { config, normalizeChatBackend } from './config.js';
import { getEngine } from './engine.js';

interface ChatMessage {
  role: string;
  content: string;
}

type BackendName = 'mock' | 'vllm' | 'agentic';

interface CompletionOptions {
  model?: string;
  messages?: ChatMessage[];
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  backend?: string;
}

interface CompletionResult {
  id: string;
  choices: Array<{
    message: { content: string; role: string };
    index: number;
    finish_reason: string;
  }>;
  model: string;
  object: string;
  created: number;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

type ProviderName = 'aurelius' | 'openai' | 'mock';

interface ProviderStat {
  requests: number;
  errors: number;
}

interface ProviderRouterOverrides {
  upstreamUrl?: string;
  vllmUpstreamUrl?: string;
  agenticUpstreamUrl?: string;
}

const AURELIUS_MODELS = [
  { id: 'aurelius-1.3b', provider: 'aurelius' },
  { id: 'aurelius-2.7b', provider: 'aurelius' },
  { id: 'aurelius-3b', provider: 'aurelius' },
  { id: 'aurelius-moe', provider: 'aurelius' },
];

const OPENAI_MODELS = [
  { id: 'gpt-4o', provider: 'openai' },
  { id: 'gpt-4o-mini', provider: 'openai' },
];

export class ProviderRouter {
  openaiApiKey: string | undefined;
  private overrides: ProviderRouterOverrides;

  private stats: Record<ProviderName, ProviderStat> = {
    aurelius: { requests: 0, errors: 0 },
    openai: { requests: 0, errors: 0 },
    mock: { requests: 0, errors: 0 },
  };

  private loadedModels = new Set<string>();

  constructor(overrides: ProviderRouterOverrides = {}) {
    this.openaiApiKey = process.env.OPENAI_API_KEY;
    this.overrides = overrides;
  }

  async complete(opts: CompletionOptions): Promise<CompletionResult & { resolved_backend?: string }> {
    const model = this.resolveModel(opts.model);
    const { backend, resolved } = this.resolveBackend(opts.backend);
    if (backend === 'mock') {
      this.stats.mock.requests++;
      return { ...(await this.completeWithMock(model, opts)), resolved_backend: resolved };
    }
    if (backend === 'vllm') {
      this.stats.aurelius.requests++;
      return { ...(await this.completeWithAurelius(this.resolveUpstreamUrl('vllm'), model, opts)), resolved_backend: resolved };
    }
    if (backend === 'agentic') {
      this.stats.aurelius.requests++;
      return { ...(await this.completeWithAurelius(this.resolveUpstreamUrl('agentic'), model, opts)), resolved_backend: resolved };
    }
    if (model.startsWith('aurelius')) {
      this.stats.aurelius.requests++;
      return { ...(await this.completeWithAurelius(this.resolveUpstreamUrl('default'), model, opts)), resolved_backend: resolved };
    }

    this.stats.openai.requests++;
    return { ...(await this.completeWithOpenAI(model, opts)), resolved_backend: resolved };
  }

  async chat(
    messages: ChatMessage[],
    options?: { model?: string; backend?: string },
  ): Promise<CompletionResult> {
    return this.complete({ messages, model: options?.model, backend: options?.backend });
  }

  getAvailableModels(): Array<{ id: string; provider: string }> {
    const models = [...AURELIUS_MODELS];
    if (this.openaiApiKey) {
      models.push(...OPENAI_MODELS);
    }
    return models;
  }

  getStats(): Record<string, ProviderStat> {
    return {
      aurelius: { ...this.stats.aurelius },
      openai: { ...this.stats.openai },
      mock: { ...this.stats.mock },
    };
  }

  loadModel(name: string): boolean {
    this.loadedModels.add(name);
    return true;
  }

  unloadModel(name: string): boolean {
    this.loadedModels.delete(name);
    return true;
  }

  resolveBackend(backend?: string): { backend: BackendName; resolved: string } {
    const fallback = normalizeChatBackend(getEngine().getConfig('chat.backend'), config.defaultChatBackend);
    if (!backend || backend === 'auto') {
      return { backend: fallback, resolved: fallback };
    }

    const normalized = normalizeChatBackend(backend, fallback);
    return { backend: normalized, resolved: normalized };
  }

  private resolveModel(model?: string): string {
    const configured = getEngine().getConfig('chat.model')?.trim();
    return model?.trim() || configured || 'aurelius-1.3b';
  }

  private resolveTemperature(temperature?: number): number {
    if (typeof temperature === 'number' && Number.isFinite(temperature)) {
      return temperature;
    }

    const configured = getEngine().getConfig('chat.temperature');
    if (configured) {
      const parsed = Number.parseFloat(configured);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }

    return 0.7;
  }

  private resolveUpstreamUrl(kind: 'default' | 'vllm' | 'agentic'): string {
    const engine = getEngine();
    const keys: Record<'default' | 'vllm' | 'agentic', string> = {
      default: 'chat.upstream_url',
      vllm: 'chat.vllm_upstream_url',
      agentic: 'chat.agentic_upstream_url',
    };
    const runtimeValue = engine.getConfig(keys[kind]);
    if (runtimeValue) {
      return runtimeValue;
    }
    if (kind === 'vllm') {
      return this.overrides.vllmUpstreamUrl || config.vllmUpstreamUrl || config.upstreamUrl;
    }
    if (kind === 'agentic') {
      return this.overrides.agenticUpstreamUrl || config.agenticUpstreamUrl || config.upstreamUrl;
    }
    return this.overrides.upstreamUrl || config.upstreamUrl;
  }

  private buildMessages(opts: CompletionOptions): ChatMessage[] {
    const messages = [...(opts.messages ?? [])];
    if (opts.prompt) {
      messages.push({ role: 'user', content: opts.prompt });
    }
    return messages;
  }

  private completeWithMock(model: string, opts: CompletionOptions): CompletionResult {
    const messages = this.buildMessages(opts);
    let lastUserMessage = '';
    for (const msg of [...messages].reverse()) {
      if (msg.role === 'user') {
        lastUserMessage = msg.content;
        break;
      }
    }
    const content = lastUserMessage
      ? `Mock response to: ${lastUserMessage}`
      : 'Mock response from Aurelius Mission Control.';
    const promptTokens = messages.reduce(
      (total, msg) => total + msg.content.split(/\s+/).filter(Boolean).length,
      0,
    );
    const completionTokens = content.split(/\s+/).filter(Boolean).length;
    return {
      id: `chatcmpl-mock-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
      },
    };
  }

  private async completeWithAurelius(
    upstreamUrl: string,
    model: string,
    opts: CompletionOptions,
  ): Promise<CompletionResult> {
    const payload = {
      model,
      messages: this.buildMessages(opts),
      max_tokens: opts.max_tokens ?? 512,
      temperature: this.resolveTemperature(opts.temperature),
      top_p: opts.top_p ?? 0.9,
      stream: Boolean(opts.stream),
    };

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (config.serviceApiKey) {
      headers.Authorization = `Bearer ${config.serviceApiKey}`;
      headers['X-Api-Key'] = config.serviceApiKey;
    }

    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 30000)
      const response = await fetch(`${upstreamUrl}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer)

      if (!response.ok) {
        throw new Error(`Aurelius upstream returned ${response.status}`);
      }

      return (await response.json()) as CompletionResult;
    } catch (error) {
      this.stats.aurelius.errors++;
      throw error;
    }
  }

  private async completeWithOpenAI(
    model: string,
    opts: CompletionOptions,
  ): Promise<CompletionResult> {
    if (!this.openaiApiKey) {
      this.stats.openai.errors++;
      throw new Error('OPENAI_API_KEY is not configured');
    }

    const payload = {
      model,
      messages: this.buildMessages(opts),
      max_tokens: opts.max_tokens ?? 512,
      temperature: this.resolveTemperature(opts.temperature),
      top_p: opts.top_p ?? 0.9,
      stream: Boolean(opts.stream),
    };

    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 30000)
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.openaiApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer)

      if (!response.ok) {
        throw new Error(`OpenAI upstream returned ${response.status}`);
      }

      return (await response.json()) as CompletionResult;
    } catch (error) {
      this.stats.openai.errors++;
      throw error;
    }
  }
}
