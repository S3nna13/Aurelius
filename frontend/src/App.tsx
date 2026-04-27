// Copyright (c) 2025 Aurelius Systems, Inc.
// Licensed under the Aurelius Open License.
// Free to use, modify, and distribute. See LICENSE for full terms.
// The Aurelius architecture remains the intellectual property of the authors.

import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ToastProvider from './components/ToastProvider';
import ErrorBoundary from './components/ErrorBoundary';
import CommandPalette from './components/CommandPalette';
import GlobalSearch from './components/GlobalSearch';
import KeyboardShortcuts from './components/KeyboardShortcuts';
import SessionLock from './components/SessionLock';
import OnboardingTour from './components/OnboardingTour';
import { AuthGuard } from './components/AuthGuard';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import Notifications from './pages/Notifications';
import Skills from './pages/Skills';
import Workflows from './pages/Workflows';
import Memory from './pages/Memory';
import Settings from './pages/Settings';
import AgentDetail from './pages/AgentDetail';
import Logs from './pages/Logs';
import ScheduledTasks from './pages/ScheduledTasks';
import AgentComparison from './pages/AgentComparison';
import HealthCheckPage from './pages/HealthCheck';
import ApiDocs from './pages/ApiDocs';
import Login from './pages/Login';
import Users from './pages/Users';
import NotFound from './pages/NotFound';
import ServerError from './pages/ServerError';
import Analytics from './pages/Analytics';
import DataExplorer from './pages/DataExplorer';
import Models from './pages/Models';
import Training from './pages/Training';
import TrainingDetail from './pages/TrainingDetail';
import Playground from './pages/Playground';

const pageVariants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
};

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        transition={{ duration: 0.18 }}
      >
        <Routes location={location}>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<AuthGuard><Dashboard /></AuthGuard>} />
          <Route path="/chat" element={<AuthGuard><Chat /></AuthGuard>} />
          <Route path="/notifications" element={<AuthGuard><Notifications /></AuthGuard>} />
          <Route path="/skills" element={<AuthGuard><Skills /></AuthGuard>} />
          <Route path="/workflows" element={<AuthGuard><Workflows /></AuthGuard>} />
          <Route path="/memory" element={<AuthGuard><Memory /></AuthGuard>} />
          <Route path="/settings" element={<AuthGuard><Settings /></AuthGuard>} />
          <Route path="/agents/:id" element={<AuthGuard><AgentDetail /></AuthGuard>} />
          <Route path="/agents" element={<AuthGuard><AgentComparison /></AuthGuard>} />
          <Route path="/logs" element={<AuthGuard><Logs /></AuthGuard>} />
          <Route path="/tasks" element={<AuthGuard><ScheduledTasks /></AuthGuard>} />
          <Route path="/health" element={<AuthGuard><HealthCheckPage /></AuthGuard>} />
          <Route path="/api-docs" element={<AuthGuard><ApiDocs /></AuthGuard>} />
          <Route path="/users" element={<AuthGuard><Users /></AuthGuard>} />
          <Route path="/analytics" element={<AuthGuard><Analytics /></AuthGuard>} />
          <Route path="/data" element={<AuthGuard><DataExplorer /></AuthGuard>} />
          <Route path="/training" element={<AuthGuard><Training /></AuthGuard>} />
          <Route path="/training/:id" element={<AuthGuard><TrainingDetail /></AuthGuard>} />
          <Route path="/models" element={<AuthGuard><Models /></AuthGuard>} />
          <Route path="/playground" element={<AuthGuard><Playground /></AuthGuard>} />
          <Route path="/500" element={<ServerError />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  );
}

function App() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const navigate = useNavigate();

  const togglePalette = useCallback(() => {
    setPaletteOpen((prev) => !prev);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        togglePalette();
        return;
      }
      if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const target = e.target as HTMLElement;
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
          return;
        }
        e.preventDefault();
        setSearchOpen(true);
        return;
      }
      // Navigation shortcuts
      if (e.key === 'g' || e.key === 'G') {
        const nextHandler = (e2: KeyboardEvent) => {
          switch (e2.key.toLowerCase()) {
            case 'd': navigate('/'); break;
            case 'c': navigate('/chat'); break;
            case 'n': navigate('/notifications'); break;
            case 's': navigate('/settings'); break;
            case 'm': navigate('/memory'); break;
            case 'k': navigate('/skills'); break;
            case 'w': navigate('/workflows'); break;
          }
          window.removeEventListener('keydown', nextHandler);
        };
        window.addEventListener('keydown', nextHandler, { once: true });
        setTimeout(() => window.removeEventListener('keydown', nextHandler), 1000);
        return;
      }
      if (e.key === 'r' || e.key === 'R') {
        if (!e.metaKey && !e.ctrlKey && !e.altKey) {
          window.dispatchEvent(new CustomEvent('aurelius:refresh'));
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePalette, navigate]);

  return (
    <ToastProvider>
      <div className="flex h-screen bg-aurelius-bg text-aurelius-text">
        <Sidebar />
        <div className="flex-1 flex flex-col md:ml-64">
          <Header onOpenPalette={togglePalette} />
          <main className="flex-1 overflow-y-auto p-6">
            <ErrorBoundary>
              <AnimatedRoutes />
            </ErrorBoundary>
          </main>
        </div>
      </div>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
      {searchOpen && <GlobalSearch onClose={() => setSearchOpen(false)} />}
      <KeyboardShortcuts />
      <SessionLock />
      <OnboardingTour />
    </ToastProvider>
  );
}

export default App;
