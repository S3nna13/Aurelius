import pino from 'pino';
import PinoHttp from 'pino-http';

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: process.env.NODE_ENV !== 'production'
    ? { target: 'pino/file', options: { destination: 1 } }
    : undefined,
  serializers: {
    req: (r: any) => ({ method: r.method, url: r.url }),
    res: (r: any) => ({ statusCode: r.statusCode }),
  },
});

export function requestLoggerMiddleware() {
  return PinoHttp({ logger, autoLogging: { ignore: (r) => (r.url ?? '').startsWith('/assets') } });
}
