export interface AgentState {
  id: string;
  state: string;
  role: string;
  metricsJson: string;
}

export interface ActivityEntry {
  id: string;
  timestamp: number;
  command: string;
  success: boolean;
  output: string;
}

export interface Notification {
  id: string;
  timestamp: number;
  channel: string;
  priority: string;
  category: string;
  title: string;
  body: string;
  read: boolean;
  delivered: boolean;
}

export interface NotificationStats {
  unread: number;
  total: number;
}

export interface MemoryLayer {
  name: string;
  entries: number;
}

export interface MemoryEntry {
  id: string;
  content: string;
  layer: string;
  timestamp: string;
  accessCount: number;
  importanceScore: number;
}

export interface LogRecord {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
}

export interface SkillEntry {
  id: string;
  name: string;
  description: string;
  scope: string;
  active: boolean;
  version: string;
  risk_score: number;
  allow_level: string;
  category: string;
}

export interface WorkflowEntry {
  id: string;
  name: string;
  status: string;
  last_run: number;
  duration: number;
  event_count: number;
}

export interface TrainingRunSummary {
  id: string;
  name: string;
  modelId: string;
  status: string;
  startedAt: number;
  currentEpoch: number;
  totalEpochs: number;
  bestValLoss: number;
  currentLr: number;
  totalSteps: number;
  dataPointCount: number;
}

export interface TrainingRunDetail extends TrainingRunSummary {
  steps: number[];
  trainLosses: number[];
  valLosses: number[];
  learningRates: number[];
  accuracies: number[];
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  path: string;
  parameterCount: number;
  state: string;
  loadedAt: string | null;
}

export interface AgentMode {
  id: string;
  name: string;
  description: string;
  allowed_tools: string[];
  response_style: string;
}

export interface StatusResponse {
  agents: AgentState[];
  skills: { id: string; active: boolean }[];
  plugins: { id: string }[];
  memory: { total_entries: number };
  notifications: { unread: number };
  counts: {
    agents_online: number;
    agents_total: number;
    skills_active: number;
    skills_total: number;
    plugins_total: number;
    notifications_unread: number;
  };
}

export interface CommandResponse {
  success: boolean;
  output: string;
  action?: string;
  target?: string;
  error?: string;
}

export interface NotificationListResponse {
  notifications: Notification[];
}

export interface MemoryResponse {
  layers: Record<string, number>;
}

export interface MemoryEntryListResponse {
  entries: MemoryEntry[];
}

export interface SkillListResponse {
  skills: SkillEntry[];
}

export interface WorkflowListResponse {
  workflows: WorkflowEntry[];
  summary: { total: number; running: number; completed: number; failed: number };
}

export interface ConfigResponse {
  config: Record<string, string>;
}

export interface ModeListResponse {
  modes: AgentMode[];
}

export interface LogListResponse {
  entries: LogRecord[];
}

export interface HealthResponse {
  status: string;
  time: number;
}

export interface LicenseResponse {
  valid: boolean;
  activated: boolean;
  tier: string;
}

export interface ActivityListResponse {
  entries: ActivityEntry[];
}

export interface WsMessage {
  channel: string;
  data: unknown;
}

export interface WsConnectedMessage {
  type: 'connected';
  heartbeatMs: number;
}

export interface WsSubscribedMessage {
  type: 'subscribed';
  channel: string;
}
