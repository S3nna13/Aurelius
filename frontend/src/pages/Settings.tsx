import { useState, useEffect, useCallback } from 'react';
import {
  Settings,
  Sliders,
  Shield,
  Server,
  Save,
  Loader2,
  AlertTriangle,
  RefreshCw,
  RotateCcw,
  Dot,
  Bell,
  Upload,
} from 'lucide-react';
import { useToast } from '../components/ToastProvider';
import ImportData from '../components/ImportData';

interface AgentMode {
  id: string;
  name: string;
  description: string;
  allowed_tools: string[];
  response_style: string;
}

interface RuntimeConfig {
  agent_mode?: string;
  log_level?: string;
  api_endpoint?: string;
  require_auth?: boolean;
  audit_logging?: boolean;
  auto_lock?: boolean;
}

const logLevels = ['debug', 'info', 'warn', 'error'];

const defaultConfig: RuntimeConfig = {
  agent_mode: 'default',
  log_level: 'info',
  api_endpoint: 'http://localhost:7870',
  require_auth: false,
  audit_logging: true,
  auto_lock: false,
};

function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

function isDirty(original: RuntimeConfig, current: RuntimeConfig): boolean {
  return JSON.stringify(original) !== JSON.stringify(current);
}

export default function SettingsPage() {
  const { toast } = useToast();
  const [modes, setModes] = useState<AgentMode[]>([]);
  const [config, setConfig] = useState<RuntimeConfig>({ ...defaultConfig });
  const [originalConfig, setOriginalConfig] = useState<RuntimeConfig>({ ...defaultConfig });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [notifPrefs, setNotifPrefs] = useState<Record<string, boolean>>({});
  const [importOpen, setImportOpen] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [modesRes, configRes, prefsRes] = await Promise.all([
        fetch('/api/modes'),
        fetch('/api/config'),
        fetch('/api/notifications/preferences'),
      ]);
      if (!modesRes.ok) throw new Error(`Modes HTTP ${modesRes.status}`);
      if (!configRes.ok) throw new Error(`Config HTTP ${configRes.status}`);
      const modesData = await modesRes.json();
      const configData = await configRes.json();
      const loadedConfig = { ...defaultConfig, ...(configData.config || {}) };
      setModes(modesData.modes || []);
      setConfig(loadedConfig);
      setOriginalConfig(loadedConfig);
      if (prefsRes.ok) {
        const prefsData = await prefsRes.json();
        setNotifPrefs(prefsData.preferences || {});
      }
      void prefsRes;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast('Failed to load settings', 'error');
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const validate = (cfg: RuntimeConfig): Record<string, string> => {
    const errs: Record<string, string> = {};
    if (cfg.api_endpoint && !isValidUrl(cfg.api_endpoint)) {
      errs.api_endpoint = 'Please enter a valid URL (e.g., http://localhost:7870)';
    }
    return errs;
  };

  const updateConfig = (updates: Partial<RuntimeConfig>) => {
    const next = { ...config, ...updates };
    setConfig(next);
    setValidationErrors(validate(next));
  };

  const resetToDefaults = () => {
    setConfig({ ...defaultConfig });
    setValidationErrors(validate(defaultConfig));
    toast('Settings reset to defaults', 'info');
  };

  const saveChanges = async () => {
    const errs = validate(config);
    if (Object.keys(errs).length > 0) {
      setValidationErrors(errs);
      toast('Please fix validation errors before saving', 'error');
      return;
    }
    setSaving(true);
    try {
      const [configRes, prefsRes] = await Promise.all([
        fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config }),
        }),
        fetch('/api/notifications/preferences', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preferences: notifPrefs }),
        }),
      ]);
      void prefsRes;
      if (!configRes.ok) throw new Error(`Config HTTP ${configRes.status}`);
      const data = await configRes.json();
      if (data.success) {
        setOriginalConfig({ ...config });
        toast('Settings saved successfully', 'success');
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const currentMode = modes.find((m) => m.id === config.agent_mode) || null;
  const dirty = isDirty(originalConfig, config);

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Settings size={20} className="text-[#4fc3f7]" />
          Settings
        </h2>
        <div className="flex gap-2 items-center">
          {dirty && (
            <span className="flex items-center gap-1 text-xs text-amber-400">
              <Dot size={16} className="animate-pulse" />
              Unsaved changes
            </span>
          )}
          <button
            onClick={() => setImportOpen(true)}
            className="aurelius-btn-outline flex items-center gap-2 text-sm"
          >
            <Upload size={14} />
            Import
          </button>
          <button
            onClick={resetToDefaults}
            disabled={loading}
            className="aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"
          >
            <RotateCcw size={14} />
            Reset
          </button>
          <button
            onClick={fetchData}
            disabled={loading}
            className="aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Refresh
          </button>
          <button
            onClick={saveChanges}
            disabled={saving || loading || Object.keys(validationErrors).length > 0}
            className="aurelius-btn flex items-center gap-2 text-sm disabled:opacity-50"
          >
            {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Save Changes
          </button>
        </div>
      </div>

      {error && (
        <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300">
          <AlertTriangle size={18} className="inline mr-2" />
          {error}
        </div>
      )}

      {loading ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
          <Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" />
          <p>Loading settings...</p>
        </div>
      ) : (
        <>
          {/* Agent Mode */}
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
              <Sliders size={16} className="text-[#4fc3f7]" />
              Agent Mode
            </h3>
            {currentMode && (
              <div className="text-xs text-[#9e9eb0] bg-[#0f0f1a] rounded-lg p-3 border border-[#2d2d44]">
                <span className="text-[#4fc3f7] font-bold">Current:</span> {currentMode.name} — {currentMode.description}
              </div>
            )}
            <div className="space-y-3">
              {modes.length === 0 ? (
                <p className="text-sm text-[#9e9eb0]">No modes available.</p>
              ) : (
                modes.map((m) => (
                  <label
                    key={m.id}
                    className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      config.agent_mode === m.id
                        ? 'border-[#4fc3f7]/40 bg-[#4fc3f7]/5'
                        : 'border-[#2d2d44]/50 bg-[#0f0f1a]/40 hover:border-[#4fc3f7]/20'
                    }`}
                  >
                    <input
                      type="radio"
                      name="agentMode"
                      value={m.id}
                      checked={config.agent_mode === m.id}
                      onChange={() => updateConfig({ agent_mode: m.id })}
                      className="mt-1 accent-[#4fc3f7]"
                    />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-[#e0e0e0]">{m.name}</p>
                      <p className="text-xs text-[#9e9eb0]">{m.description}</p>
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {m.allowed_tools.map((t) => (
                          <span
                            key={t}
                            className="text-[10px] bg-[#2d2d44]/30 text-[#9e9eb0] px-1.5 py-0.5 rounded border border-[#2d2d44]/40"
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                  </label>
                ))
              )}
            </div>
          </div>

          {/* System Config */}
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
              <Server size={16} className="text-[#4fc3f7]" />
              System Configuration
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-[#9e9eb0] mb-1.5">API Endpoint</label>
                <input
                  type="text"
                  value={config.api_endpoint || ''}
                  onChange={(e) => updateConfig({ api_endpoint: e.target.value })}
                  className={`w-full bg-[#0f0f1a] border rounded-lg px-3 py-2 text-sm text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7] ${
                    validationErrors.api_endpoint ? 'border-rose-500' : 'border-[#2d2d44]'
                  }`}
                />
                {validationErrors.api_endpoint && (
                  <p className="text-xs text-rose-400 mt-1">{validationErrors.api_endpoint}</p>
                )}
              </div>
              <div>
                <label className="block text-xs text-[#9e9eb0] mb-1.5">Log Level</label>
                <select
                  value={config.log_level || 'info'}
                  onChange={(e) => updateConfig({ log_level: e.target.value })}
                  className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7]"
                >
                  {logLevels.map((level) => (
                    <option key={level} value={level}>
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Notification Preferences */}
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
              <Bell size={16} className="text-[#4fc3f7]" />
              Notification Preferences
            </h3>
            <div className="space-y-3">
              {['agent', 'system', 'alerts'].map((channel) => (
                <label key={channel} className="flex items-center justify-between py-2">
                  <span className="text-sm text-[#e0e0e0] capitalize">{channel} notifications</span>
                  <input
                    type="checkbox"
                    checked={notifPrefs[channel] !== false}
                    onChange={(e) =>
                      setNotifPrefs((prev) => ({ ...prev, [channel]: e.target.checked }))
                    }
                    className="accent-[#4fc3f7] w-4 h-4"
                  />
                </label>
              ))}
            </div>
          </div>

          {/* Security */}
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
              <Shield size={16} className="text-[#4fc3f7]" />
              Security
            </h3>
            <div className="space-y-3">
              <label className="flex items-center justify-between py-2">
                <span className="text-sm text-[#e0e0e0]">Require auth for local API</span>
                <input
                  type="checkbox"
                  checked={!!config.require_auth}
                  onChange={(e) => updateConfig({ require_auth: e.target.checked })}
                  className="accent-[#4fc3f7] w-4 h-4"
                />
              </label>
              <label className="flex items-center justify-between py-2">
                <span className="text-sm text-[#e0e0e0]">Enable audit logging</span>
                <input
                  type="checkbox"
                  checked={!!config.audit_logging}
                  onChange={(e) => updateConfig({ audit_logging: e.target.checked })}
                  className="accent-[#4fc3f7] w-4 h-4"
                />
              </label>
              <label className="flex items-center justify-between py-2">
                <span className="text-sm text-[#e0e0e0]">Auto-lock after inactivity</span>
                <input
                  type="checkbox"
                  checked={!!config.auto_lock}
                  onChange={(e) => {
                    updateConfig({ auto_lock: e.target.checked });
                    localStorage.setItem('aurelius-auto-lock', String(e.target.checked));
                  }}
                  className="accent-[#4fc3f7] w-4 h-4"
                />
              </label>
            </div>
          </div>
        </>
      )}
      {importOpen && (
        <ImportData
          onClose={() => setImportOpen(false)}
          onImport={(data) => {
            if (typeof data === 'object' && data !== null && 'config' in data) {
              setConfig({ ...defaultConfig, ...(data as any).config });
            }
          }}
          title="Import Settings"
        />
      )}
    </div>
  );
}
