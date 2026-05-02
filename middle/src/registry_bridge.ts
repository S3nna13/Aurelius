import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

export interface RegistryAgent {
  id: string;
  name: string;
  description: string;
  category: string;
  capabilities: string[];
  default_tools: string[];
  icon: string;
  color: string;
  parameters: Record<string, unknown>;
}

export interface RegistrySkill {
  id: string;
  name: string;
  description: string;
  category: string;
  agent_types: string[];
  tags: string[];
}

export interface RegistrySnapshot {
  agents: RegistryAgent[];
  agent_categories: string[];
  skills: RegistrySkill[];
  skill_categories: string[];
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

let cachedSnapshot: RegistrySnapshot | null = null;
let pendingSnapshot: Promise<RegistrySnapshot> | null = null;

function runPythonJson(code: string): Promise<RegistrySnapshot> {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', ['-c', code], { cwd: repoRoot });
    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    python.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    python.on('error', reject);
    python.on('close', (exitCode) => {
      if (exitCode !== 0) {
        reject(new Error(stderr.trim() || `python3 exited with code ${exitCode}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout) as RegistrySnapshot);
      } catch (error) {
        reject(new Error(`Failed to parse registry snapshot: ${(error as Error).message}`));
      }
    });
  });
}

export async function loadRegistrySnapshot(forceRefresh = false): Promise<RegistrySnapshot> {
  if (cachedSnapshot && !forceRefresh) {
    return cachedSnapshot;
  }

  if (pendingSnapshot && !forceRefresh) {
    return pendingSnapshot;
  }

  const code = `
import json
import sys

sys.path.insert(0, '.')

from aurelius.api_registry import get_registry_snapshot

print(json.dumps(get_registry_snapshot()))
`;

  pendingSnapshot = runPythonJson(code).then((snapshot) => {
    cachedSnapshot = snapshot;
    pendingSnapshot = null;
    return snapshot;
  });

  return pendingSnapshot;
}
