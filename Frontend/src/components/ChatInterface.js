import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { FaPaperPlane, FaSpinner, FaUser, FaRobot, FaExpand } from 'react-icons/fa';
import { sendMessage } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Welcome to HomeoAI Assistant! 🌿\n\nI'm here to help you with homeopathic information. How can I assist you today?"
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await sendMessage(userMessage);
      setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "I apologize, but I'm having trouble connecting to the server. Please try again later."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="chat-header-content">
          <FaRobot className="chat-icon" />
          <div>
            <h2>HomeoAI Chat</h2>
            <p className="subtitle">Ask anything about homeopathy</p>
          </div>
        </div>
        <button className="expand-button" title="Full screen">
          <FaExpand />
        </button>
      </div>

      <div className="messages-container">
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`message-wrapper ${message.role === 'user' ? 'user' : 'assistant'}`}
            >
              <div className="message-avatar">
                {message.role === 'user' ? <FaUser /> : <FaRobot />}
              </div>
              <div className="message-bubble">
                <ReactMarkdown>
                  {message.content}
                </ReactMarkdown>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="message-wrapper assistant"
          >
            <div className="message-avatar">
              <FaRobot />
            </div>
            <div className="message-bubble loading">
              <div className="loading-spinner">
                <div className="spinner-container">
                  <div className="spinner-dot"></div>
                  <div className="spinner-dot"></div>
                  <div className="spinner-dot"></div>
                </div>
                <span className="loading-text">Thinking...</span>
              </div>
            </div>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your question here..."
            rows={1}
            className="message-input"
            disabled={isLoading}
          />
          <button
            className={`send-button ${!input.trim() || isLoading ? 'disabled' : ''}`}
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
          >
            {isLoading ? <FaSpinner className="spinner" /> : <FaPaperPlane />}
          </button>
        </div>
        <div className="input-hints">
          <span>Press Enter to send, Shift+Enter for new line</span>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;