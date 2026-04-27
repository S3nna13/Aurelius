import { type ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

interface AuthGuardProps {
  children: ReactNode
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { authenticated } = useAuth()
  const location = useLocation()

  if (!authenticated && location.pathname !== '/login') {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}
