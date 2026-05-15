import React, { useState, useEffect } from "react";

interface HardwareInfo {
  cpu_arch: string;
  total_ram_gb: number;
  gpu_vram_gb: number;
  unified_memory: boolean;
  gpu_name: string;
  cuda_available: boolean;
  mlx_available: boolean;
}

interface MemoryBudget {
  total_memory_gb: number;
  available_for_aurelius_gb: number;
  used_gb: number;
  free_gb: number;
  pressure_level: string;
}

interface ProfileRecommendation {
  profile: string;
  recommended_models: Record<string, string>;
}

const HardwareDashboard: React.FC = () => {
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [memory, setMemory] = useState<MemoryBudget | null>(null);
  const [profile, setProfile] = useState<ProfileRecommendation | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/hardware/detect")
      .then(r => r.json())
      .then(data => {
        setHardware(data.info);
        setProfile({ profile: data.profile, recommended_models: data.recommended_models });
        setLoading(false);
      })
      .catch(() => setLoading(false));

    fetch("/api/health")
      .then(r => r.json())
      .then(data => setMemory(data.memory))
      .catch(() => {});
  }, []);

  const pressureColor = (level: string) => {
    switch (level) {
      case "low": return "#22c55e";
      case "moderate": return "#eab308";
      case "high": return "#f97316";
      case "critical": return "#ef4444";
      case "emergency": return "#dc2626";
      default: return "#94a3b8";
    }
  };

  if (loading) return <div className="p-8">Detecting hardware...</div>;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Hardware Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-2">CPU & Memory</h2>
          <p className="text-gray-400">Architecture: {hardware?.cpu_arch || "unknown"}</p>
          <p className="text-gray-400">Total RAM: {hardware?.total_ram_gb || "?"} GB</p>
          <p className="text-gray-400">Unified Memory: {hardware?.unified_memory ? "Yes" : "No"}</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-2">GPU</h2>
          <p className="text-gray-400">Device: {hardware?.gpu_name || "None detected"}</p>
          <p className="text-gray-400">VRAM: {hardware?.gpu_vram_gb || 0} GB</p>
          <p className="text-gray-400">CUDA: {hardware?.cuda_available ? "Yes" : "No"}</p>
          <p className="text-gray-400">MLX: {hardware?.mlx_available ? "Yes" : "No"}</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-2">Profile</h2>
          <p className="text-gray-400">Recommended: {profile?.profile || "unknown"}</p>
          {profile && Object.entries(profile.recommended_models || {}).map(([m, a]) => (
            <p key={m} className="text-gray-400 capitalize">{m}: {a}</p>
          ))}
        </div>
      </div>

      {memory && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-3">Memory Budget</h2>
          <div className="w-full bg-gray-700 rounded-full h-8 relative">
            <div
              className="h-8 rounded-full transition-all duration-500"
              style={{
                width: `${(memory.used_gb / memory.available_for_aurelius_gb) * 100}%`,
                backgroundColor: pressureColor(memory.pressure_level),
              }}
            />
            <span className="absolute inset-0 flex items-center justify-center text-sm font-medium text-gray-100">
              {memory.used_gb.toFixed(1)} / {memory.available_for_aurelius_gb.toFixed(1)} GB ({((memory.used_gb / memory.available_for_aurelius_gb) * 100).toFixed(0)}%)
            </span>
          </div>
          <p className="mt-2 text-gray-400">
            Pressure: <span style={{ color: pressureColor(memory.pressure_level) }} className="font-semibold capitalize">{memory.pressure_level}</span>
            {" "}| Available: {memory.free_gb.toFixed(1)} GB
          </p>
        </div>
      )}
    </div>
  );
};

export default HardwareDashboard;
