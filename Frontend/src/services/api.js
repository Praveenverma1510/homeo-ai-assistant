const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const sendMessage = async (message, history = []) => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, history }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to send message');
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const analyzeSymptoms = async (symptoms) => {
  try {
    const response = await fetch(`${API_BASE_URL}/symptoms`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symptoms }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to analyze symptoms');
    }

    const data = await response.json();
    return data.analysis;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const searchRemedies = async (query) => {
  try {
    const response = await fetch(`${API_BASE_URL}/remedies?q=${encodeURIComponent(query)}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to search remedies');
    }

    const data = await response.json();
    return data.remedies;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    return data.status === 'healthy';
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};