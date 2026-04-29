import { config } from './config.js';

interface ChatMessage {
  role: string;
  content: string;
}

interface CompletionOptions {
  model?: string;
  messages?: ChatMessage[];
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
}

interface CompletionResult {
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

type ProviderName = 'aurelius' | 'openai';

interface ProviderStat {
  requests: number;
  errors: number;
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

  private stats: Record<ProviderName, ProviderStat> = {
    aurelius: { requests: 0, errors: 0 },
    openai: { requests: 0, errors: 0 },
  };

  private loadedModels = new Set<string>();

  constructor() {
    this.openaiApiKey = process.env.OPENAI_API_KEY;
  }

  async complete(opts: CompletionOptions): Promise<CompletionResult> {
    const model = opts.model || 'aurelius-1.3b';
    if (model.startsWith('aurelius')) {
      this.stats.aurelius.requests++;
      return this.completeWithAurelius(model, opts);
    }

    this.stats.openai.requests++;
    return this.completeWithOpenAI(model, opts);
  }

  async chat(messages: ChatMessage[], options?: { model?: string }): Promise<CompletionResult> {
    return this.complete({ messages, model: options?.model });
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

  private buildMessages(opts: CompletionOptions): ChatMessage[] {
    const messages = [...(opts.messages ?? [])];
    if (opts.prompt) {
      messages.push({ role: 'user', content: opts.prompt });
    }
    return messages;
  }

  private async completeWithAurelius(
    model: string,
    opts: CompletionOptions,
  ): Promise<CompletionResult> {
    const payload = {
      model,
      messages: this.buildMessages(opts),
      max_tokens: opts.max_tokens ?? 512,
      temperature: opts.temperature ?? 0.7,
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
      const response = await fetch(`${config.upstreamUrl}/v1/chat/completions`, {
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
      temperature: opts.temperature ?? 0.7,
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
