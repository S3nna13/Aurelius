import { useState, useEffect } from 'react';
import {
  Settings,
  Sliders,
  Shield,
  Server,
  Save,
  Loader2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { useToast } from '../components/ToastProvider';

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

export default function SettingsPage() {
  const { toast } = useToast();
  const [modes, setModes] = useState<AgentMode[]>([]);
  const [config, setConfig] = useState<RuntimeConfig>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modesRes, configRes] = await Promise.all([
        fetch('/api/modes'),
        fetch('/api/config'),
      ]);
      if (!modesRes.ok) throw new Error(`Modes HTTP ${modesRes.status}`);
      if (!configRes.ok) throw new Error(`Config HTTP ${configRes.status}`);
      const modesData = await modesRes.json();
      const configData = await configRes.json();
      setModes(modesData.modes || []);
      setConfig(configData.config || {});
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast('Failed to load settings', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const updateConfig = (updates: Partial<RuntimeConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  };

  const saveChanges = async () => {
    setSaving(true);
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.success) {
        toast('Settings saved successfully', 'success');
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const currentMode = modes.find((m) => m.id === config.agent_mode) || null;

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Settings size={20} className="text-[#4fc3f7]" />
          Settings
        </h2>
        <div className="flex gap-2">
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
            disabled={saving || loading}
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
                  className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7]"
                />
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
                  onChange={(e) => updateConfig({ auto_lock: e.target.checked })}
                  className="accent-[#4fc3f7] w-4 h-4"
                />
              </label>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
