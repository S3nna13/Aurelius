import { accessSync, constants } from 'fs';
import { resolve } from 'path';

function isExecutable(candidate: string): boolean {
  try {
    accessSync(candidate, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

export function resolvePythonExecutable(repoRoot: string): string {
  const envOverride = process.env.AURELIUS_PYTHON?.trim() || process.env.PYTHON?.trim();
  if (envOverride) {
    return envOverride;
  }

  const candidates = [
    resolve(repoRoot, '.venv', 'bin', 'python3.14'),
    resolve(repoRoot, '.venv', 'bin', 'python3'),
    resolve(repoRoot, '.venv', 'bin', 'python'),
    resolve(repoRoot, '.venv', 'Scripts', 'python.exe'),
  ];

  for (const candidate of candidates) {
    if (isExecutable(candidate)) {
      return candidate;
    }
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}