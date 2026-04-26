import { useState, useCallback } from 'react';

interface UseApiOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: unknown;
}

interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  execute: (opts?: UseApiOptions) => Promise<T | null>;
}

export function useApi<T = unknown>(path: string): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(
    async (opts?: UseApiOptions): Promise<T | null> => {
      setLoading(true);
      setError(null);
      try {
        const url = path.startsWith('/api/') ? path : `/api${path}`;
        const response = await fetch(url, {
          method: opts?.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
            ...opts?.headers,
          },
          body: opts?.body ? JSON.stringify(opts.body) : undefined,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const json = (await response.json()) as T;
        setData(json);
        return json;
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err));
        setError(e);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [path]
  );

  return { data, loading, error, execute };
}
