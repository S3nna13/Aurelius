import React, { useState, useEffect } from "react";

const DAIESDashboard: React.FC = () => {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const runGates = () => {
    setLoading(true);
    fetch("/api/daies/run", { method: "POST" })
      .then(r => r.json())
      .then(data => { setResults(data); setLoading(false); })
      .catch(() => setLoading(false));
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">DAIES Validation Gates</h1>
        <button onClick={runGates} disabled={loading} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white disabled:opacity-50">
          {loading ? "Running..." : "Run All Gates"}
        </button>
      </div>

      {results && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-blue-400">{results.total}</div>
            <div className="text-gray-400 text-sm">Total Checked</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-green-400">{results.passed}</div>
            <div className="text-gray-400 text-sm">Passed</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-red-400">{results.failed}</div>
            <div className="text-gray-400 text-sm">Failed</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-gray-200">{results.discovered}</div>
            <div className="text-gray-400 text-sm">Skills Discovered</div>
          </div>
        </div>
      )}

      {results && results.gates && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-3">Gate Results</h2>
          <table className="w-full text-sm text-gray-300">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left">Gate</th>
                <th className="text-center">Passed</th>
                <th className="text-center">Failed</th>
                <th className="text-center">Rate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results.gates).map(([name, data]: [string, any]) => (
                <tr key={name} className="border-b border-gray-800">
                  <td className="py-2 font-mono text-xs">{name}</td>
                  <td className="text-center text-green-400">{data.passed}</td>
                  <td className="text-center text-red-400">{data.failed}</td>
                  <td className="text-center">{data.passed + data.failed > 0 ? Math.round(data.passed / (data.passed + data.failed) * 100) : 0}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default DAIESDashboard;
