import React from "react";

const Settings: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Settings</h1>

      <div className="bg-gray-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-gray-200">General</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Preferred Model</label>
            <select className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100">
              <option value="swift">Aurelius Swift</option>
              <option value="forge" selected>Aurelius Forge</option>
              <option value="atlas">Aurelius Atlas</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Backend</label>
            <select className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100">
              <option value="mlx">MLX (Apple Silicon)</option>
              <option value="tensorrt_llm">TensorRT-LLM (CUDA)</option>
              <option value="llama_cpp">llama.cpp (GGUF)</option>
              <option value="remote">Remote Aurelius</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Memory Policy</label>
            <select className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100">
              <option value="conservative">Conservative</option>
              <option value="balanced" selected>Balanced</option>
              <option value="performance">Performance</option>
              <option value="frontier">Frontier</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Approval Mode</label>
            <select className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100">
              <option value="inline" selected>Inline</option>
              <option value="batch">Batch</option>
              <option value="always">Always</option>
              <option value="never">Never</option>
            </select>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-gray-200">Remote Endpoint</h2>
        <div className="flex gap-4">
          <input className="flex-1 p-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100" placeholder="https://your-aurelius-remote.example.com" />
          <button className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white">Save</button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
