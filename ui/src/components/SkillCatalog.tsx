import React, { useState, useEffect } from "react";

interface SkillInfo {
  id: string;
  name: string;
  version: string;
  category: string;
  summary: string;
  risk_level: string;
  status: string;
  permissions: Array<{ name: string; scope: string }>;
  tags: string[];
}

interface SkillStats {
  total_skills: number;
  loaded_skills: number;
  by_category: Record<string, number>;
}

const SkillCatalog: React.FC = () => {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [stats, setStats] = useState<SkillStats | null>(null);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/skills/native")
      .then(r => r.json())
      .then(data => {
        setSkills(data.skills);
        setStats(data.stats);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filtered = skills.filter(s =>
    !filter ||
    s.category.toLowerCase().includes(filter.toLowerCase()) ||
    s.name.toLowerCase().includes(filter.toLowerCase()) ||
    s.id.toLowerCase().includes(filter.toLowerCase())
  );

  if (loading) return <div className="p-8">Loading skills...</div>;

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Native Skill Catalog</h1>
        <span className="text-gray-400">{stats?.total_skills || 0} skills | {stats ? Object.keys(stats.by_category).length : 0} categories</span>
      </div>

      <input
        type="text"
        className="w-full p-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-100"
        placeholder="Filter by category, name, or ID..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
      />

      {stats && (
        <div className="flex gap-2 flex-wrap">
          {Object.entries(stats.by_category).map(([cat, count]) => (
            <button
              key={cat}
              className="px-3 py-1 bg-gray-700 rounded-full text-sm text-gray-300 hover:bg-gray-600"
              onClick={() => setFilter(cat)}
            >
              {cat} ({count})
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map(skill => (
          <div key={skill.id} className="bg-gray-800 rounded-lg p-4 hover:border-blue-500 border border-transparent transition-all">
            <h3 className="font-semibold text-gray-100">{skill.name}</h3>
            <p className="text-xs text-gray-500 font-mono">{skill.id}</p>
            <p className="text-sm text-gray-400 mt-2">{skill.summary}</p>
            <div className="mt-3 flex gap-2 items-center">
              <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                skill.risk_level === "critical" ? "bg-red-900 text-red-200" :
                skill.risk_level === "high" ? "bg-orange-900 text-orange-200" :
                skill.risk_level === "medium" ? "bg-yellow-900 text-yellow-200" :
                "bg-green-900 text-green-200"
              }`}>
                {skill.risk_level}
              </span>
              <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300 capitalize">{skill.category}</span>
            </div>
            {skill.tags.length > 0 && (
              <div className="mt-2 flex gap-1 flex-wrap">
                {skill.tags.slice(0, 4).map(t => (
                  <span key={t} className="px-1.5 py-0.5 bg-gray-900 rounded text-xs text-gray-500">#{t}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SkillCatalog;
