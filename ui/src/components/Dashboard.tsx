import React, { useState, useEffect } from "react";

interface HealthData {
  status: string;
  version: string;
  hardware: {
    cpu_arch: string;
    total_ram_gb: number;
    gpu_vram_gb: number;
    unified_memory: boolean;
    cuda_available: boolean;
    mlx_available: boolean;
  };
  memory: {
    pressure_level: string;
    used_gb: number;
    free_gb: number;
  };
}

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/health")
      .then(r => r.json())
      .then(data => { setHealth(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-gray-400">Loading dashboard...</div>;
  if (!health) return <div className="p-8 text-red-400">Could not load health data.</div>;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Mission Control Dashboard</h1>

      <div className="flex gap-4">
        <div className="flex-1 bg-gray-800 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-green-400">{health.hardware.total_ram_gb} GB</div>
          <div className="text-gray-400 text-sm">Total RAM</div>
        </div>
        <div className="flex-1 bg-gray-800 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-blue-400">{health.hardware.gpu_vram_gb} GB</div>
          <div className="text-gray-400 text-sm">GPU VRAM</div>
        </div>
        <div className="flex-1 bg-gray-800 rounded-lg p-4 text-center">
          <div className={`text-3xl font-bold capitalize ${
            health.memory.pressure_level === "low" ? "text-green-400" :
            health.memory.pressure_level === "moderate" ? "text-yellow-400" : "text-red-400"
          }`}>{health.memory.pressure_level}</div>
          <div className="text-gray-400 text-sm">Memory Pressure</div>
        </div>
        <div className="flex-1 bg-gray-800 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-gray-200">{health.version || "v2"}</div>
          <div className="text-gray-400 text-sm">Version</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-2">System Status</h2>
          <div className="space-y-2 text-sm text-gray-300">
            <p>Status: <span className={health.status === "healthy" ? "text-green-400" : "text-red-400"}>{health.status}</span></p>
            <p>CPU: {health.hardware.cpu_arch}</p>
            <p>Unified Memory: {health.hardware.unified_memory ? "Yes" : "No"}</p>
            <p>CUDA: {health.hardware.cuda_available ? "Available" : "Not available"}</p>
            <p>MLX: {health.hardware.mlx_available ? "Available" : "Not available"}</p>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-2">Memory Usage</h2>
          <div className="space-y-2 text-sm text-gray-300">
            <p>Used: {health.memory.used_gb.toFixed(1)} GB</p>
            <p>Free: {health.memory.free_gb.toFixed(1)} GB</p>
            <p>Pressure: <span className="capitalize">{health.memory.pressure_level}</span></p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
