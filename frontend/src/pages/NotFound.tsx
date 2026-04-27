import { Link } from 'react-router-dom'
import { Home, ArrowLeft, Search } from 'lucide-react'

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="text-center space-y-6 max-w-md">
        <div className="w-24 h-24 mx-auto rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
          <Search size={48} className="text-amber-400" />
        </div>
        <div>
          <h1 className="text-6xl font-bold text-[#e0e0e0]">404</h1>
          <p className="text-lg text-[#9e9eb0] mt-2">Page not found</p>
          <p className="text-sm text-[#9e9eb0]/60 mt-1">The page you're looking for doesn't exist or has been moved.</p>
        </div>
        <div className="flex items-center justify-center gap-3">
          <button onClick={() => window.history.back()} className="aurelius-btn-outline flex items-center gap-2 text-sm">
            <ArrowLeft size={14} /> Go Back
          </button>
          <Link to="/" className="aurelius-btn flex items-center gap-2 text-sm">
            <Home size={14} /> Dashboard
          </Link>
        </div>
      </div>
    </div>
  )
}
