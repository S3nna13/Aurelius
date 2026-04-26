export class ApiError extends Error {
  status: number;
  cause: unknown;

  constructor(status: number, message: string, cause?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.cause = cause;
  }
}

export class AuthError extends ApiError {
  constructor(message = 'Authentication required') {
    super(401, message);
    this.name = 'AuthError';
  }
}

export class NetworkError extends Error {
  cause: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = 'NetworkError';
    this.cause = cause;
  }
}

export class TimeoutError extends NetworkError {
  constructor(url: string, timeoutMs: number) {
    super(`Request to ${url} timed out after ${timeoutMs}ms`);
    this.name = 'TimeoutError';
  }
}
