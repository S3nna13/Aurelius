import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ToastProvider from './components/ToastProvider';
import ErrorBoundary from './components/ErrorBoundary';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import Notifications from './pages/Notifications';
import Skills from './pages/Skills';
import Workflows from './pages/Workflows';
import Memory from './pages/Memory';
import Settings from './pages/Settings';

function App() {
  return (
    <ToastProvider>
      <div className="flex h-screen bg-aurelius-bg text-aurelius-text">
        <Sidebar />
        <div className="flex-1 flex flex-col md:ml-64">
          <Header />
          <main className="flex-1 overflow-y-auto p-6">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/chat" element={<Chat />} />
                <Route path="/notifications" element={<Notifications />} />
                <Route path="/skills" element={<Skills />} />
                <Route path="/workflows" element={<Workflows />} />
                <Route path="/memory" element={<Memory />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>
      </div>
    </ToastProvider>
  );
}

export default App;
