import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import ChatInterface from './components/ChatInterface';
import SymptomAnalyzer from './components/SymptomAnalyzer';
import RemedySearch from './components/RemedySearch';
import { FaBrain, FaHeartbeat, FaSearch, FaHome } from 'react-icons/fa';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check backend health
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/health');
        const data = await response.json();
        if (data.status === 'healthy') {
          setIsConnected(true);
        }
      } catch (error) {
        console.error('Backend not reachable:', error);
        setIsConnected(false);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  const tabs = [
    { id: 'chat', label: 'Chat', icon: FaBrain },
    { id: 'symptoms', label: 'Analyze Symptoms', icon: FaHeartbeat },
    { id: 'remedies', label: 'Search Remedies', icon: FaSearch },
  ];

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="logo-section"
          >
            <FaHome className="logo-icon" />
            <h1>HomeoAI Assistant</h1>
            <span className="badge">v1.0</span>
          </motion.div>
          <div className="status-indicator">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
            <span className="status-text">
              {loading ? 'Connecting...' : isConnected ? 'Connected' : 'Offline'}
            </span>
          </div>
        </div>
      </header>

      <nav className="tab-navigation">
        <div className="tab-container">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <motion.button
                key={tab.id}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Icon className="tab-icon" />
                <span>{tab.label}</span>
              </motion.button>
            );
          })}
        </div>
      </nav>

      <main className="App-main">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="tab-content"
        >
          {activeTab === 'chat' && <ChatInterface />}
          {activeTab === 'symptoms' && <SymptomAnalyzer />}
          {activeTab === 'remedies' && <RemedySearch />}
        </motion.div>
      </main>

    </div>
  );
}

export default App;