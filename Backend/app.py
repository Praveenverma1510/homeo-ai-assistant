import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from utils.rag_pipeline import HomeoRAGPipeline
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])  # React dev server

# Initialize RAG pipeline
rag_pipeline = None

def initialize_rag():
    """Initialize the RAG pipeline with error handling"""
    global rag_pipeline
    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = HomeoRAGPipeline()
        rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        return jsonify({
            'status': 'error',
            'message': 'RAG pipeline not initialized'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model': 'ArogyaAI-LLaMA3-8B',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message parameter'
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'error': 'Empty message'
            }), 400
        
        # Get history if provided
        history = data.get('history', [])
        
        # Process through RAG pipeline
        logger.info(f"Processing message: {user_message[:50]}...")
        response = rag_pipeline.generate_response(user_message, history)
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/symptoms', methods=['POST'])
def analyze_symptoms():
    """Endpoint for symptom analysis"""
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({
                'error': 'Missing symptoms parameter'
            }), 400
        
        symptoms = data['symptoms'].strip()
        if not symptoms:
            return jsonify({
                'error': 'Empty symptoms'
            }), 400
        
        # Create a specialized prompt for symptom analysis
        prompt = f"""Based on the following symptoms, please analyze and suggest possible homeopathic remedies.
        Provide a detailed analysis including:
        1. Symptom pattern recognition
        2. Possible constitutional types
        3. Top 3 remedy suggestions with indications
        4. General recommendations
        
        Symptoms: {symptoms}
        
        Please note: This is for educational purposes only. Always consult a qualified homeopath."""
        
        response = rag_pipeline.generate_response(prompt, [])
        
        return jsonify({
            'analysis': response,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in symptom analysis: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/remedies', methods=['GET'])
def search_remedies():
    """Search for specific remedies"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({
                'error': 'Missing search query'
            }), 400
        
        # Use retriever directly for remedy search
        results = rag_pipeline.retriever.get_relevant_documents(query)
        
        remedies = []
        for doc in results[:5]:  # Limit to 5 results
            remedies.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return jsonify({
            'remedies': remedies,
            'count': len(remedies),
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in remedy search: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize RAG before starting server
    if initialize_rag():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize RAG pipeline. Exiting...")
        exit(1)