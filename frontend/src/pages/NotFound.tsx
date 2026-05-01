import { useNavigate } from 'react-router-dom';
import { Home, Search, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  const navigate = useNavigate();
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="text-6xl font-bold text-[#2d2d44] mb-4">404</div>
        <h1 className="text-xl font-bold text-[#e0e0e0] mb-2">Page Not Found</h1>
        <p className="text-sm text-[#9e9eb0] mb-6">The page you're looking for doesn't exist or has been moved.</p>
        <div className="flex justify-center gap-3">
          <button onClick={() => navigate(-1)} className="aurelius-btn-outline flex items-center gap-2 text-sm"><ArrowLeft size={14} /> Go Back</button>
          <button onClick={() => navigate('/')} className="aurelius-btn-primary flex items-center gap-2 text-sm"><Home size={14} /> Dashboard</button>
        </div>
      </div>
    </div>
  );
}
