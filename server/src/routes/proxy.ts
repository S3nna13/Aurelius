import type { Request, Response } from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { loadConfig } from '../config.js';

const config = loadConfig();

export const proxyToPython = createProxyMiddleware({
  target: config.pythonUrl,
  changeOrigin: true,
  pathFilter: '/api/chat',
  pathRewrite: { '^/api/chat': '/v1/chat/completions' },
  timeout: 120_000,
  proxyTimeout: 120_000,
  on: {
    error(err, _req, res) {
      const srvRes = res as unknown as Response;
      srvRes.statusCode = 502;
      srvRes.end(JSON.stringify({ error: 'Upstream unavailable', message: err.message }));
    },
  },
});
