// Copyright (c) 2025 Aurelius Systems, Inc.
// Licensed under the Aurelius Open License.
// Free to use, modify, and distribute. See LICENSE for full terms.
// The Aurelius architecture remains the intellectual property of the authors.

import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { lazy, Suspense, useState, useEffect, useCallback } from 'react';
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

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Chat = lazy(() => import('./pages/Chat'));
const Notifications = lazy(() => import('./pages/Notifications'));
const Skills = lazy(() => import('./pages/Skills'));
const Workflows = lazy(() => import('./pages/Workflows'));
const Memory = lazy(() => import('./pages/Memory'));
const Settings = lazy(() => import('./pages/Settings'));
const AgentDetail = lazy(() => import('./pages/AgentDetail'));
const Logs = lazy(() => import('./pages/Logs'));
const ScheduledTasks = lazy(() => import('./pages/ScheduledTasks'));
const AgentComparison = lazy(() => import('./pages/AgentComparison'));
const HealthCheckPage = lazy(() => import('./pages/HealthCheck'));
const ApiDocs = lazy(() => import('./pages/ApiDocs'));
const Models = lazy(() => import('./pages/Models'));
const Training = lazy(() => import('./pages/Training'));

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
        <Suspense fallback={<div className="flex items-center justify-center p-12 text-aurelius-muted">Loading...</div>}>
          <Routes location={location}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/notifications" element={<Notifications />} />
            <Route path="/skills" element={<Skills />} />
            <Route path="/workflows" element={<Workflows />} />
            <Route path="/memory" element={<Memory />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/agents/:id" element={<AgentDetail />} />
            <Route path="/agents" element={<AgentComparison />} />
            <Route path="/logs" element={<Logs />} />
            <Route path="/tasks" element={<ScheduledTasks />} />
            <Route path="/health" element={<HealthCheckPage />} />
            <Route path="/api-docs" element={<ApiDocs />} />
            <Route path="/models" element={<Models />} />
            <Route path="/training" element={<Training />} />
            <Route path="/training/:id" element={<Training />} />
          </Routes>
        </Suspense>
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
