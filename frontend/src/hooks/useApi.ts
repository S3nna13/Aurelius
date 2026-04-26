// Copyright (c) 2025 Aurelius Systems, Inc.
// Licensed under the Aurelius Open License.
// Free to use, modify, and distribute. See LICENSE for full terms.
// The Aurelius architecture remains the intellectual property of the authors.

import { useState, useCallback, useRef, useEffect } from 'react';

interface UseApiOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: unknown;
}

interface UseApiConfig {
  refreshInterval?: number;
  retries?: number;
  timeout?: number;
}

interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  execute: (opts?: UseApiOptions) => Promise<T | null>;
  refresh: () => Promise<T | null>;
}

const DEFAULT_TIMEOUT = 10000;
const DEFAULT_RETRIES = 2;

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

async function fetchWithRetry(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  retries: number
): Promise<Response> {
  let lastErr: Error | undefined;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetchWithTimeout(url, init, timeoutMs);
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err));
      if (attempt < retries) {
        const delay = Math.min(1000 * 2 ** attempt, 5000);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastErr;
}

export function useApi<T = unknown>(
  path: string,
  config: UseApiConfig = {}
): UseApiResult<T> {
  const { refreshInterval, retries = DEFAULT_RETRIES, timeout = DEFAULT_TIMEOUT } = config;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const execute = useCallback(
    async (opts?: UseApiOptions): Promise<T | null> => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
      const controller = new AbortController();
      abortRef.current = controller;

      setLoading(true);
      setError(null);
      try {
        const url = path.startsWith('/api/') ? path : `/api${path}`;
        const apiKey = localStorage.getItem('aurelius-api-key') || '';
        const response = await fetchWithRetry(
          url,
          {
            method: opts?.method || 'GET',
            headers: {
              'Content-Type': 'application/json',
              ...(apiKey ? { 'X-API-Key': apiKey } : {}),
              ...opts?.headers,
            },
            body: opts?.body ? JSON.stringify(opts.body) : undefined,
          },
          timeout,
          retries
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const json = (await response.json()) as T;
        if (!controller.signal.aborted) {
          setData(json);
        }
        return json;
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err));
        if (!controller.signal.aborted) {
          setError(e);
        }
        return null;
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    },
    [path, timeout, retries]
  );

  const refresh = useCallback(() => execute(), [execute]);

  useEffect(() => {
    if (!refreshInterval || refreshInterval <= 0) return;
    const checkEnabled = () => {
      try {
        return localStorage.getItem('aurelius-auto-refresh') !== 'false';
      } catch {
        return true;
      }
    };
    const id = setInterval(() => {
      if (checkEnabled()) refresh();
    }, refreshInterval);
    const handleToggle = (e: Event) => {
      const enabled = (e as CustomEvent).detail;
      if (enabled) refresh();
    };
    window.addEventListener('aurelius:auto-refresh', handleToggle);
    return () => {
      clearInterval(id);
      window.removeEventListener('aurelius:auto-refresh', handleToggle);
    };
  }, [refreshInterval, refresh]);

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  return { data, loading, error, execute, refresh };
}
