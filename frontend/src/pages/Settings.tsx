import { Settings, Sliders, Shield, Server, Save } from 'lucide-react';
import { useState } from 'react';

const agentModes = [
  { key: 'autonomous', label: 'Autonomous', description: 'Agents act independently with full permissions.' },
  { key: 'supervised', label: 'Supervised', description: 'Agents request approval before sensitive actions.' },
  { key: 'manual', label: 'Manual', description: 'All actions require explicit user confirmation.' },
];

export default function SettingsPage() {
  const [mode, setMode] = useState('supervised');
  const [apiEndpoint, setApiEndpoint] = useState('http://localhost:8080');
  const [logLevel, setLogLevel] = useState('info');

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
          <Settings size={20} className="text-aurelius-accent" />
          Settings
        </h2>
        <button className="aurelius-btn flex items-center gap-2 text-sm">
          <Save size={14} />
          Save Changes
        </button>
      </div>

      {/* Agent Mode */}
      <div className="aurelius-card space-y-4">
        <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
          <Sliders size={16} className="text-aurelius-accent" />
          Agent Mode
        </h3>
        <div className="space-y-3">
          {agentModes.map((m) => (
            <label
              key={m.key}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                mode === m.key
                  ? 'border-aurelius-accent/40 bg-aurelius-accent/5'
                  : 'border-aurelius-border/50 bg-aurelius-bg/40 hover:border-aurelius-accent/20'
              }`}
            >
              <input
                type="radio"
                name="agentMode"
                value={m.key}
                checked={mode === m.key}
                onChange={() => setMode(m.key)}
                className="mt-1 accent-aurelius-accent"
              />
              <div>
                <p className="text-sm font-medium text-aurelius-text">{m.label}</p>
                <p className="text-xs text-aurelius-muted">{m.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* System Config */}
      <div className="aurelius-card space-y-4">
        <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
          <Server size={16} className="text-aurelius-accent" />
          System Configuration
        </h3>
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-aurelius-muted mb-1.5">API Endpoint</label>
            <input
              type="text"
              value={apiEndpoint}
              onChange={(e) => setApiEndpoint(e.target.value)}
              className="w-full bg-aurelius-bg border border-aurelius-border rounded-lg px-3 py-2 text-sm text-aurelius-text focus:outline-none focus:border-aurelius-accent"
            />
          </div>
          <div>
            <label className="block text-xs text-aurelius-muted mb-1.5">Log Level</label>
            <select
              value={logLevel}
              onChange={(e) => setLogLevel(e.target.value)}
              className="w-full bg-aurelius-bg border border-aurelius-border rounded-lg px-3 py-2 text-sm text-aurelius-text focus:outline-none focus:border-aurelius-accent"
            >
              <option value="debug">Debug</option>
              <option value="info">Info</option>
              <option value="warn">Warning</option>
              <option value="error">Error</option>
            </select>
          </div>
        </div>
      </div>

      {/* Security */}
      <div className="aurelius-card space-y-4">
        <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
          <Shield size={16} className="text-aurelius-accent" />
          Security
        </h3>
        <div className="space-y-3">
          <label className="flex items-center justify-between py-2">
            <span className="text-sm text-aurelius-text">Require auth for local API</span>
            <input type="checkbox" defaultChecked className="accent-aurelius-accent w-4 h-4" />
          </label>
          <label className="flex items-center justify-between py-2">
            <span className="text-sm text-aurelius-text">Enable audit logging</span>
            <input type="checkbox" defaultChecked className="accent-aurelius-accent w-4 h-4" />
          </label>
          <label className="flex items-center justify-between py-2">
            <span className="text-sm text-aurelius-text">Auto-lock after inactivity</span>
            <input type="checkbox" className="accent-aurelius-accent w-4 h-4" />
          </label>
        </div>
      </div>
    </div>
  );
}
