import React, { useState } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { FaHeartbeat, FaStethoscope, FaSpinner, FaClipboardList } from 'react-icons/fa';
import { analyzeSymptoms } from '../services/api';
import './SymptomAnalyzer.css';

const SymptomAnalyzer = () => {
  const [symptoms, setSymptoms] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const commonSymptoms = [
    'Headache', 'Fever', 'Cough', 'Fatigue', 'Nausea',
    'Anxiety', 'Insomnia', 'Joint Pain', 'Skin Rash', 'Digestive Issues'
  ];

  const handleAnalyze = async () => {
    if (!symptoms.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      const result = await analyzeSymptoms(symptoms);
      setAnalysis(result);
    } catch (error) {
      setError(error.message || 'Failed to analyze symptoms. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSymptomClick = (symptom) => {
    setSymptoms(prev => prev ? `${prev}, ${symptom}` : symptom);
  };

  return (
    <div className="symptom-analyzer">
      <div className="analyzer-header">
        <div className="header-content">
          <FaHeartbeat className="header-icon" />
          <div>
            <h2>Symptom Analyzer</h2>
            <p className="subtitle">Describe your symptoms for a homeopathic analysis</p>
          </div>
        </div>
      </div>

      <div className="analyzer-body">
        <div className="input-section">
          <div className="symptom-input-wrapper">
            <textarea
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              placeholder="Describe your symptoms in detail... (e.g., 'I have a throbbing headache on the right side, worse in the morning, and I feel irritable')"
              rows={4}
              className="symptom-textarea"
              disabled={isLoading}
            />
            <div className="input-actions">
              <button
                className={`analyze-button ${!symptoms.trim() || isLoading ? 'disabled' : ''}`}
                onClick={handleAnalyze}
                disabled={!symptoms.trim() || isLoading}
              >
                {isLoading ? (
                  <>
                    <FaSpinner className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <FaStethoscope />
                    Analyze Symptoms
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="common-symptoms">
            <p className="label">Common symptoms to select from:</p>
            <div className="symptom-tags">
              {commonSymptoms.map((symptom) => (
                <motion.button
                  key={symptom}
                  className="symptom-tag"
                  onClick={() => handleSymptomClick(symptom)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  disabled={isLoading}
                >
                  {symptom}
                </motion.button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="error-message"
          >
            <FaClipboardList className="error-icon" />
            <span>{error}</span>
          </motion.div>
        )}

        {analysis && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="analysis-results"
          >
            <div className="analysis-header">
              <FaClipboardList className="analysis-icon" />
              <h3>Analysis Results</h3>
            </div>
            <div className="analysis-content">
              <ReactMarkdown
                components={{
                  h1: ({ children, ...props }) => <h1 className="analysis-h1" {...props}>{children}</h1>,
                  h2: ({ children, ...props }) => <h2 className="analysis-h2" {...props}>{children}</h2>,
                  h3: ({ children, ...props }) => <h3 className="analysis-h3" {...props}>{children}</h3>,
                  ul: ({ children, ...props }) => <ul className="analysis-ul" {...props}>{children}</ul>,
                  ol: ({ children, ...props }) => <ol className="analysis-ol" {...props}>{children}</ol>,
                  li: ({ children, ...props }) => <li className="analysis-li" {...props}>{children}</li>,
                  strong: ({ children, ...props }) => <strong className="analysis-strong" {...props}>{children}</strong>,
                  p: ({ children, ...props }) => <p className="analysis-p" {...props}>{children}</p>,
                  blockquote: ({ children, ...props }) => <blockquote className="analysis-blockquote" {...props}>{children}</blockquote>,
                }}
              >
                {analysis}
              </ReactMarkdown>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default SymptomAnalyzer;