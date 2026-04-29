import { useState } from 'react';
import { Users, Shield, Key, MoreVertical, Search, Plus } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';

interface User { id: string; name: string; email: string; role: string; status: string; lastActive: string; }

export default function UsersPage() {
  const { data } = useApi<{ users: User[] }>('/auth/users');
  const [search, setSearch] = useState('');
  const users = (data?.users || []).filter(u => !search || u.name.toLowerCase().includes(search.toLowerCase()) || u.email.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Users size={20} className="text-[#4fc3f7]" />Users</h2>
        <button className="aurelius-btn-primary flex items-center gap-2 text-sm"><Plus size={14} />Invite User</button>
      </div>
      <div className="relative w-64"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search users..." className="pl-8 py-1.5 text-sm" /></div>
      <div className="space-y-2">
        {users.map(user => (
          <div key={user.id} className="aurelius-card p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-[#4fc3f7]/10 flex items-center justify-center"><Users size={14} className="text-[#4fc3f7]" /></div>
              <div><p className="text-sm font-medium text-[#e0e0e0]">{user.name}</p><p className="text-xs text-[#9e9eb0]">{user.email}</p></div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-[#9e9eb0]">{user.role}</span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full ${user.status === 'active' ? 'text-emerald-400 bg-emerald-500/10' : 'text-[#9e9eb0] bg-[#0f0f1a]'}`}>{user.status}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
