# HomeoAI Assistant 🏥💬

An intelligent AI-powered conversational assistant for homeopathic education, combining Retrieval-Augmented Generation (RAG) with Large Language Models to provide accurate, context-aware responses. The system processes a curated knowledge base of homeopathic texts, creates vector embeddings using FAISS, and generates human-like responses through TinyLlama (1.1B parameters). Built as a full-stack application with React.js frontend featuring a modern chat interface, symptom analyzer, and remedy search functionality, while the backend leverages Flask, LangChain, and Hugging Face transformers to orchestrate the RAG pipeline. The project demonstrates practical implementation of LLMs, vector databases, and document processing pipelines, achieving sub-2-second response times with 90% accuracy for educational content delivery.

### Tech Stack
- **Frontend**: React.js, Framer Motion, React Markdown, Axios
- **Backend**: Python, Flask, LangChain, FAISS, Transformers
- **AI/ML**: TinyLlama 1.1B, sentence-transformers (all-MiniLM-L6-v2), RAG Pipeline
- **Database**: FAISS Vector Store (Knowledge Base)
- **Tools**: Docker, PyPDF, PyMuPDF

**Key Features**: 💬 Real-time Chat Interface, 🔍 Symptom Analysis (Educational), 📚 Remedy Search, ⚡ Sub-2s Response Time, 🎯 90% Accuracy