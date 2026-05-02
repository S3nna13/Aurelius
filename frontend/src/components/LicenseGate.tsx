// Copyright (c) 2025 Aurelius Systems, Inc.
// Licensed under the Aurelius Open License.
// Free to use, modify, and distribute. See LICENSE for full terms.
// The Aurelius architecture remains the intellectual property of the authors.

import { useState, useEffect } from 'react';
import { Shield, KeyRound, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';

interface LicenseGateProps {
  children: React.ReactNode;
}

export default function LicenseGate({ children }: LicenseGateProps) {
  const [checking, setChecking] = useState(true);
  const [valid, setValid] = useState(false);
  const [key, setKey] = useState('');
  const [activating, setActivating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const storedKey = localStorage.getItem('aurelius-api-key');
    if (!storedKey) {
      setChecking(false);
      return;
    }
    fetch('/api/license/validate', {
      headers: { 'X-API-Key': storedKey },
    })
      .then((res) => res.json())
      .then((data) => {
        setValid(data.valid === true);
        setChecking(false);
      })
      .catch(() => {
        setValid(false);
        setChecking(false);
        setError('Network error during validation. Click to retry.');
      });
  }, []);

  const activate = async () => {
    if (!key.trim()) return;
    setActivating(true);
    setError(null);
    try {
      const res = await fetch('/api/license/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ license_key: key.trim() }),
      });
      const data = await res.json();
      if (data.success) {
        localStorage.setItem('aurelius-api-key', key.trim());
        localStorage.setItem('aurelius-license-tier', data.tier);
        setValid(true);
      } else {
        setError(data.error || 'Activation failed');
      }
    } catch {
      setError('Network error during activation');
    } finally {
      setActivating(false);
    }
  };

  if (checking) {
    return (
      <div className="flex h-screen items-center justify-center bg-aurelius-bg text-aurelius-text">
        <div className="flex flex-col items-center gap-3">
          <Loader2 size={32} className="animate-spin text-aurelius-accent" />
          <p className="text-sm text-aurelius-muted">Validating license...</p>
        </div>
      </div>
    );
  }

  if (!valid) {
    return (
      <div className="flex h-screen items-center justify-center bg-aurelius-bg text-aurelius-text p-4">
        <div className="bg-aurelius-card border border-aurelius-border rounded-xl p-8 max-w-md w-full shadow-2xl">
          <div className="flex flex-col items-center gap-4 mb-6">
            <div className="w-16 h-16 rounded-full bg-aurelius-accent/10 border border-aurelius-accent/30 flex items-center justify-center">
              <Shield size={32} className="text-aurelius-accent" />
            </div>
            <h1 className="text-xl font-bold text-aurelius-text">Aurelius</h1>
            <p className="text-sm text-aurelius-muted text-center">
              License activation required. Please enter your license key to continue.
            </p>
          </div>

          <div className="space-y-4">
            <div className="relative">
              <KeyRound size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-aurelius-muted" />
              <input
                type="text"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && activate()}
                placeholder="AURELIUS-XXXXXXXXXXXXXXXX"
                className="w-full bg-aurelius-bg border border-aurelius-border rounded-lg pl-10 pr-4 py-3 text-sm text-aurelius-text placeholder:text-aurelius-muted focus:outline-none focus:border-aurelius-accent font-mono"
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-500/5 border border-rose-500/20 rounded-lg px-3 py-2">
                <AlertTriangle size={14} />
                {error}
              </div>
            )}

            <button
              onClick={activate}
              disabled={activating || !key.trim()}
              className="aurelius-btn w-full flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {activating ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <CheckCircle2 size={16} />
              )}
              {activating ? 'Activating...' : 'Activate License'}
            </button>
          </div>

          <p className="text-[10px] text-aurelius-muted text-center mt-6">
            By activating, you agree to the Aurelius EULA and Proprietary License.
          </p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
