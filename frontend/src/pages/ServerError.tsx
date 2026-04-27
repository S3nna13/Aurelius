import { Link } from 'react-router-dom'
import { Home, RefreshCw, AlertTriangle } from 'lucide-react'

export default function ServerError() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="text-center space-y-6 max-w-md">
        <div className="w-24 h-24 mx-auto rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
          <AlertTriangle size={48} className="text-rose-400" />
        </div>
        <div>
          <h1 className="text-6xl font-bold text-[#e0e0e0]">500</h1>
          <p className="text-lg text-[#9e9eb0] mt-2">Server error</p>
          <p className="text-sm text-[#9e9eb0]/60 mt-1">Something went wrong on our end. Please try again later.</p>
        </div>
        <div className="flex items-center justify-center gap-3">
          <button onClick={() => window.location.reload()} className="aurelius-btn-outline flex items-center gap-2 text-sm">
            <RefreshCw size={14} /> Retry
          </button>
          <Link to="/" className="aurelius-btn flex items-center gap-2 text-sm">
            <Home size={14} /> Dashboard
          </Link>
        </div>
      </div>
    </div>
  )
}
