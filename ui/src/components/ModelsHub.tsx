import React, { useState } from "react";

interface ModelInfo {
  name: string;
  size: string;
  purpose: string;
  target: string;
  status: string;
}

const ModelsHub: React.FC = () => {
  const [models] = useState<ModelInfo[]>([
    { name: "Aurelius Swift", size: "~0.6B", purpose: "Edge, router, verifier", target: "Jetson Nano+, Mac 8GB+", status: "Ready" },
    { name: "Aurelius Forge", size: "~3B", purpose: "Default agent, coding, CUA", target: "Mac 16GB+, RTX 8-24GB+", status: "Ready" },
    { name: "Aurelius Atlas", size: "~32B/8B active", purpose: "Frontier reasoning, orchestration", target: "RTX 6000+, Blackwell, Mac Ultra", status: "Ready" },
  ]);

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold text-gray-100">Model Hub</h1>
      <table className="w-full text-sm text-gray-300">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left py-2">Model</th>
            <th className="text-left py-2">Size</th>
            <th className="text-left py-2">Purpose</th>
            <th className="text-left py-2">Target Hardware</th>
            <th className="text-left py-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {models.map(m => (
            <tr key={m.name} className="border-b border-gray-800 hover:bg-gray-750">
              <td className="py-3 font-semibold text-gray-100">{m.name}</td>
              <td>{m.size}</td>
              <td>{m.purpose}</td>
              <td>{m.target}</td>
              <td><span className="px-2 py-1 bg-green-900 text-green-200 rounded text-xs">{m.status}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ModelsHub;
