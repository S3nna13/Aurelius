import type {
  HealthResponse,
  StatusResponse,
  CommandResponse,
  ActivityListResponse,
  NotificationListResponse,
  NotificationStats,
  MemoryResponse,
  MemoryEntryListResponse,
  SkillListResponse,
  WorkflowListResponse,
  ConfigResponse,
  ModeListResponse,
  LogListResponse,
  LicenseResponse,
  AgentState,
  ModelInfo,
  SkillEntry,
  WorkflowEntry,
  TrainingRunDetail,
  TrainingRunSummary,
} from './types';
import { ApiError, AuthError, NetworkError, TimeoutError } from './errors';

export interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
}

const DEFAULT_TIMEOUT = 10_000;
const DEFAULT_RETRIES = 2;

export class AureliusClient {
  private baseUrl: string;
  private apiKey: string;
  private timeout: number;
  private retries: number;

  constructor(config: ClientConfig = {}) {
    this.baseUrl = config.baseUrl || '';
    this.apiKey = config.apiKey || (typeof localStorage !== 'undefined' ? localStorage.getItem('aurelius-api-key') || '' : '');
    this.timeout = config.timeout || DEFAULT_TIMEOUT;
    this.retries = config.retries ?? DEFAULT_RETRIES;
  }

  setApiKey(key: string): void {
    this.apiKey = key;
  }

  getApiKey(): string {
    return this.apiKey;
  }

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}/api${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.retries; attempt++) {
      try {
        const headers: Record<string, string> = {
          'Content-Type': 'application/json',
          ...(options.headers as Record<string, string> || {}),
        };
        if (this.apiKey) headers['X-API-Key'] = this.apiKey;

        const res = await fetch(url, {
          ...options,
          headers,
          signal: controller.signal,
        });

        if (!res.ok) {
          if (res.status === 401) throw new AuthError();
          const body = await res.json().catch(() => ({}));
          throw new ApiError(res.status, body.message || body.error || res.statusText);
        }

        return (await res.json()) as T;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        if (lastError instanceof ApiError) throw lastError;
        if (lastError.name === 'AbortError') throw new TimeoutError(url, this.timeout);
        if (attempt < this.retries) {
          await new Promise(r => setTimeout(r, Math.min(1000 * 2 ** attempt, 5000)));
        }
      } finally {
        clearTimeout(timeoutId);
      }
    }

    throw lastError || new NetworkError(`Request to ${url} failed`);
  }

  // Health
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  // Status
  async getStatus(): Promise<StatusResponse> {
    return this.request<StatusResponse>('/status');
  }

  async getAgent(id: string): Promise<AgentState> {
    return this.request<AgentState>(`/agents/${id}`);
  }

  // Command
  async postCommand(command: string): Promise<CommandResponse> {
    return this.request<CommandResponse>('/command', {
      method: 'POST',
      body: JSON.stringify({ command }),
    });
  }

  // Activity
  async getActivity(limit?: number): Promise<ActivityListResponse> {
    const params = limit ? `?limit=${limit}` : '';
    return this.request<ActivityListResponse>(`/activity${params}`);
  }

  // Notifications
  async getNotifications(params?: {
    category?: string;
    priority?: string;
    read?: boolean;
    limit?: number;
  }): Promise<NotificationListResponse> {
    const qs = new URLSearchParams();
    if (params?.category) qs.set('category', params.category);
    if (params?.priority) qs.set('priority', params.priority);
    if (params?.read !== undefined) qs.set('read', String(params.read));
    if (params?.limit) qs.set('limit', String(params.limit));
    const query = qs.toString();
    return this.request<NotificationListResponse>(`/notifications${query ? `?${query}` : ''}`);
  }

  async getNotificationStats(): Promise<NotificationStats> {
    return this.request<NotificationStats>('/notifications/stats');
  }

  async markNotificationRead(id: string): Promise<{ success: boolean }> {
    return this.request('/notifications/read', {
      method: 'POST',
      body: JSON.stringify({ id }),
    });
  }

  async markAllNotificationsRead(category?: string): Promise<{ success: boolean; count: number }> {
    return this.request('/notifications/read-all', {
      method: 'POST',
      body: JSON.stringify({ category }),
    });
  }

  async getNotificationPrefs(): Promise<{ preferences: Record<string, boolean> }> {
    return this.request('/notifications/preferences');
  }

  async setNotificationPrefs(preferences: Record<string, boolean>): Promise<{ success: boolean }> {
    return this.request('/notifications/preferences', {
      method: 'POST',
      body: JSON.stringify({ preferences }),
    });
  }

  // Skills
  async getSkills(): Promise<SkillListResponse> {
    return this.request<SkillListResponse>('/skills');
  }

  async getSkillDetail(id: string): Promise<Record<string, unknown>> {
    return this.request(`/skills/${id}`);
  }

  async executeSkill(skillId: string, variables?: Record<string, unknown>): Promise<{ success: boolean; output: string; duration_ms: number }> {
    return this.request('/skills/execute', {
      method: 'POST',
      body: JSON.stringify({ skill_id: skillId, variables }),
    });
  }

  // Workflows
  async getWorkflows(): Promise<WorkflowListResponse> {
    return this.request<WorkflowListResponse>('/workflows');
  }

  async getWorkflowDetail(id: string): Promise<Record<string, unknown>> {
    return this.request(`/workflows/${id}`);
  }

  async triggerWorkflow(id: string, trigger: string): Promise<{ success: boolean; workflow_id: string; trigger: string; state: string }> {
    return this.request(`/workflows/${id}/trigger`, {
      method: 'POST',
      body: JSON.stringify({ trigger }),
    });
  }

  // Memory
  async getMemoryLayers(): Promise<MemoryResponse> {
    return this.request<MemoryResponse>('/memory');
  }

  async getMemoryEntries(params?: { layer?: string; q?: string; limit?: number }): Promise<MemoryEntryListResponse> {
    const qs = new URLSearchParams();
    if (params?.layer) qs.set('layer', params.layer);
    if (params?.q) qs.set('q', params.q);
    if (params?.limit) qs.set('limit', String(params.limit));
    const query = qs.toString();
    return this.request<MemoryEntryListResponse>(`/memory/entries${query ? `?${query}` : ''}`);
  }

  // Config
  async getConfig(): Promise<ConfigResponse> {
    return this.request<ConfigResponse>('/config');
  }

  async setConfig(config: Record<string, string>): Promise<{ success: boolean; config: Record<string, string> }> {
    return this.request('/config', {
      method: 'POST',
      body: JSON.stringify({ config }),
    });
  }

  // Modes
  async getModes(): Promise<ModeListResponse> {
    return this.request<ModeListResponse>('/modes');
  }

  // Logs
  async getLogs(params?: { level?: string; q?: string; limit?: number }): Promise<LogListResponse> {
    const qs = new URLSearchParams();
    if (params?.level) qs.set('level', params.level);
    if (params?.q) qs.set('q', params.q);
    if (params?.limit) qs.set('limit', String(params.limit));
    const query = qs.toString();
    return this.request<LogListResponse>(`/logs${query ? `?${query}` : ''}`);
  }

  // License
  async validateLicense(): Promise<LicenseResponse> {
    return this.request<LicenseResponse>('/license/validate');
  }

  async activateLicense(licenseKey: string, tier?: string): Promise<{ success: boolean; tier: string }> {
    return this.request('/license/activate', {
      method: 'POST',
      body: JSON.stringify({ license_key: licenseKey, tier }),
    });
  }

  // Generic helpers
  async get<T = unknown>(path: string, params?: Record<string, string>): Promise<T> {
    const qs = params ? `?${new URLSearchParams(params)}` : '';
    return this.request<T>(`${path}${qs}`);
  }

  async post<T = unknown>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  // Models
  async listModels(): Promise<{ models: ModelInfo[] }> {
    return this.request('/models');
  }

  async getModel(id: string): Promise<ModelInfo> {
    return this.request(`/models/${id}`);
  }

  async setModelState(id: string, state: string): Promise<{ success: boolean; model: ModelInfo }> {
    return this.post(`/models/${id}/state`, { state });
  }

  // Skills (engine-backed)
  async listSkillsEngine(): Promise<{ skills: SkillEntry[] }> {
    return this.request('/skills');
  }

  // Workflows (engine-backed)
  async listWorkflowsEngine(): Promise<{ workflows: WorkflowEntry[]; summary: { total: number; running: number; completed: number; failed: number } }> {
    return this.request('/workflows');
  }

  async updateWorkflowStatus(id: string, trigger: string): Promise<{ success: boolean; workflow_id: string; trigger: string; state: string }> {
    return this.post(`/workflows/${id}/trigger`, { trigger });
  }

  // Training
  async listTrainingRuns(): Promise<{ runs: TrainingRunSummary[]; summary: { total: number; running: number; completed: number; queued: number } }> {
    return this.request('/training');
  }

  async getTrainingRun(id: string): Promise<TrainingRunDetail> {
    return this.request(`/training/${id}`);
  }

  async createTrainingRun(name: string, model_id: string, total_epochs?: number): Promise<{ success: boolean; run: TrainingRunSummary }> {
    return this.post('/training', { name, model_id, total_epochs });
  }
}

export const api = new AureliusClient();
