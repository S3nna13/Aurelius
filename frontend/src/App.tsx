import { Routes, Route, useLocation } from 'react-router-dom';
import { useState, useEffect, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ToastProvider from './components/ToastProvider';
import ErrorBoundary from './components/ErrorBoundary';
import CommandPalette from './components/CommandPalette';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import Notifications from './pages/Notifications';
import Skills from './pages/Skills';
import Workflows from './pages/Workflows';
import Memory from './pages/Memory';
import Settings from './pages/Settings';

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
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/skills" element={<Skills />} />
          <Route path="/workflows" element={<Workflows />} />
          <Route path="/memory" element={<Memory />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  );
}

function App() {
  const [paletteOpen, setPaletteOpen] = useState(false);

  const togglePalette = useCallback(() => {
    setPaletteOpen((prev) => !prev);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        togglePalette();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePalette]);

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
    </ToastProvider>
  );
}

export default App;
