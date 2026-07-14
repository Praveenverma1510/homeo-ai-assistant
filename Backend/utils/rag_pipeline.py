import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    GenerationConfig
)
import torch

logger = logging.getLogger(__name__)


class HomeoRAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline with configuration"""
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.retriever = None
        self.embeddings = None

        # Memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Configuration
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.k_retrieval = 3

        # Data paths
        self.data_path = "data/homeopathy_data.txt"
        self.index_path = "faiss_homeo_index"

        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = self.device == "cuda"

        # System prompt for the model
        self.system_prompt = """You are a knowledgeable and empathetic AI assistant specializing in homeopathy. 
        You can have friendly conversations while also providing accurate homeopathic information when asked.
        
        For greetings and casual conversation:
        - Respond warmly and naturally
        - Ask how you can help
        - Stay within your role as a homeopathy assistant
        
        For homeopathy questions:
        - Provide accurate information based on homeopathic principles
        - Include the law of similars, individualization, and minimum dose
        - Suggest remedies with key indications
        - Always include a medical disclaimer
        
        For symptom analysis:
        - Ask clarifying questions about the symptoms
        - Suggest possible constitutional types
        - Recommend remedies with key indications
        - Include a medical disclaimer"""

        logger.info(f"Using device: {self.device}")

    def initialize(self) -> bool:
        """Initialize all components of the RAG pipeline"""
        try:
            logger.info("=" * 50)
            logger.info("Initializing HomeoAI RAG Pipeline")
            logger.info("=" * 50)

            # Step 1: Load and process documents
            logger.info("Step 1: Loading documents...")
            documents = self.load_documents()

            if not documents:
                logger.warning("No documents found. Creating sample data...")
                self.create_sample_data()
                documents = self.load_documents()

            if not documents:
                raise ValueError("No documents could be loaded")

            logger.info(f"✓ Loaded {len(documents)} documents")

            # Step 2: Create embeddings
            logger.info("Step 2: Creating embeddings...")
            self.embeddings = self.create_embeddings()
            logger.info("✓ Embeddings model loaded")

            # Step 3: Create vector store
            logger.info("Step 3: Creating vector store...")
            self.vectorstore = self.create_vectorstore(documents)
            logger.info("✓ Vector store created")

            # Step 4: Initialize LLM
            logger.info("Step 4: Initializing LLM...")
            self.llm = self.initialize_llm()
            logger.info("✓ LLM initialized")

            # Step 5: Create retriever
            logger.info("Step 5: Creating retriever...")
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k_retrieval}
            )
            logger.info("✓ Retriever created")

            # Step 6: Create QA chain
            logger.info("Step 6: Creating QA chain...")
            self.qa_chain = self.create_qa_chain()
            logger.info("✓ QA chain created")

            logger.info("=" * 50)
            logger.info("✅ RAG Pipeline initialized successfully!")
            logger.info("=" * 50)
            return True

        except Exception as e:
            logger.error(f"❌ Error initializing RAG pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load_documents(self) -> List:
        """Load and split documents from various sources"""
        documents = []

        # Try loading from text file
        if os.path.exists(self.data_path):
            try:
                loader = TextLoader(self.data_path, encoding='utf-8')
                docs = loader.load()
                if docs:
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from text file")
            except Exception as e:
                logger.warning(f"Error loading text file: {str(e)}")

        # Try loading from PDFs
        pdf_dir = "data/pdfs"
        if os.path.exists(pdf_dir):
            for pdf_file in os.listdir(pdf_dir):
                if pdf_file.endswith('.pdf'):
                    try:
                        loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
                        docs = loader.load()
                        if docs:
                            documents.extend(docs)
                            logger.info(
                                f"Loaded {len(docs)} documents from PDF: {pdf_file}")
                    except Exception as e:
                        logger.warning(
                            f"Error loading PDF {pdf_file}: {str(e)}")

        if not documents:
            logger.warning("No documents loaded from any source")
            return []

        # Split documents into chunks
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",  # Paragraphs
                    "\n",    # Lines
                    ". ",    # Sentences
                    "! ",    # Exclamations
                    "? ",    # Questions
                    "; ",    # Semicolons
                    ", ",    # Commas
                    " ",     # Words
                    ""       # Characters
                ],
                length_function=len,
                keep_separator=True
            )

            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents")

            # Log sample chunk
            if chunks:
                sample = chunks[0].page_content[:200].replace('\n', ' ')
                logger.info(f"Sample chunk: {sample}...")

            return chunks

        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return []

    def create_sample_data(self):
        """Create comprehensive sample homeopathy data"""
        os.makedirs("data", exist_ok=True)

        sample_data = """# Homeopathy Materia Medica - Complete Reference

## Aconitum Napellus (Aconite)
**Mental Symptoms:**
- Great anxiety, fear, and restlessness
- Fear of death, predicts time of death
- Panic attacks with palpitations

**Physical Symptoms:**
- Sudden onset of symptoms
- High fever with dry, hot skin
- Thirst for cold water
- Numbness and tingling

**Modalities:**
- Worse: Evening and night, dry cold winds
- Better: Open air, warm room

**Key Indications:**
- Acute inflammations
- Croup, bronchitis
- Fever with restlessness
- Neuralgia with numbness

## Belladonna
**Mental Symptoms:**
- Delirium, rage, biting, striking
- Hallucinations, sees terrifying images
- Violence, wants to escape

**Physical Symptoms:**
- Red face, hot head, cold extremities
- Dilated pupils, photophobia
- Throbbing headaches
- Dry mouth with thirst

**Modalities:**
- Worse: Touch, jar, light, at 3 PM
- Better: Rest, warmth

**Key Indications:**
- Acute inflammations
- Scarlet fever
- Meningitis
- Sunstroke

## Nux Vomica
**Mental Symptoms:**
- Irritable, oversensitive, angry
- Cannot bear contradiction
- Hypersensitive to noise, light, odors

**Physical Symptoms:**
- Constipation with ineffectual urging
- Digestive issues from overeating
- Headaches from mental exertion
- Spasmodic conditions

**Modalities:**
- Worse: Morning, cold, mental exertion
- Better: Rest, warmth, pressure

**Key Indications:**
- Digestive disorders
- Hangovers
- Spasmodic conditions
- Insomnia from mental work

## Pulsatilla
**Mental Symptoms:**
- Weepy, whining, changeable mood
- Needs sympathy, consolation
- Mild, yielding disposition

**Physical Symptoms:**
- Thick, greenish, bland discharges
- Thirstless, dry mouth
- Symptoms change rapidly
- Hormonal imbalances

**Modalities:**
- Worse: Warm rooms, rich fatty food
- Better: Open air, walking slowly

**Key Indications:**
- Women's disorders
- Catarrhal conditions
- Digestive issues
- Hormonal imbalances

## Arsenicum Album
**Mental Symptoms:**
- Great anxiety, restlessness
- Fear of death, of being alone
- Fastidious, meticulous

**Physical Symptoms:**
- Burning pains relieved by heat
- Thirst for small quantities often
- Skin dry, scaly, burning
- Debility with restlessness

**Modalities:**
- Worse: Cold, at 12-2 AM
- Better: Warm drinks, warmth

**Key Indications:**
- Food poisoning
- Anxiety disorders
- Chronic skin conditions
- Asthma

## Sulphur
**Mental Symptoms:**
- Selfish, philosophical, filthy
- Aversion to bathing, washing
- Discontented with everything

**Physical Symptoms:**
- Burning soles, offensive discharges
- Hungry at 11 AM
- Skin conditions with itching
- Hemorrhoids

**Modalities:**
- Worse: Warmth, standing
- Better: Dry weather, open air

**Key Indications:**
- Chronic skin conditions
- Hemorrhoids
- Morning diarrhea
- Respiratory conditions

## Lycopodium
**Mental Symptoms:**
- Anxious, cowardly, apprehensive
- Fear of public speaking
- Anticipation anxiety

**Physical Symptoms:**
- Bloating after eating
- Headaches from hunger
- Right-sided symptoms
- Digestive issues

**Modalities:**
- Worse: 4-8 PM, warm food
- Better: Motion, cold drinks

**Key Indications:**
- Digestive disorders
- Respiratory conditions
- Liver issues
- Anxiety disorders

## Ignatia Amara
**Mental Symptoms:**
- Hysterical, sighing, sobbing
- Contradictory symptoms
- Grief, shock, disappointment

**Physical Symptoms:**
- Globus hystericus
- Headaches from grief
- Spasmodic conditions
- Insomnia

**Modalities:**
- Worse: Coffee, tobacco, emotions
- Better: Distraction, warmth

**Key Indications:**
- Grief, shock
- Emotional disturbances
- Spasmodic cough
- Insomnia

## Sepia
**Mental Symptoms:**
- Indifferent, irritable
- Aversion to family and friends
- Weeps when telling symptoms

**Physical Symptoms:**
- Bearing down sensation in pelvis
- Worse before menses
- Hormonal imbalances
- Skin conditions

**Modalities:**
- Worse: Cold, at 3 PM
- Better: Violent exercise

**Key Indications:**
- Hormonal imbalances
- Menopausal symptoms
- Depression
- Skin conditions

## Natrum Muriaticum
**Mental Symptoms:**
- Reserved, brooding, sensitive
- Aversion to consolation
- Holds grief inside

**Physical Symptoms:**
- Craving for salt
- Dry mucous membranes
- Headaches from sun
- Skin conditions

**Modalities:**
- Worse: Sun, mental exertion
- Better: Open air, fasting

**Key Indications:**
- Emotional disorders
- Skin conditions
- Allergies
- Headaches

# Homeopathic Principles

## Law of Similars
"Like cures like" - A substance that can cause symptoms in a healthy person can cure similar symptoms in a sick person. This is the fundamental principle of homeopathy.

## Minimum Dose
The smallest dose that can produce a therapeutic effect should be used. Homeopathic remedies are diluted to minimize toxicity while maintaining therapeutic action.

## Individualization
Treatment should be tailored to the individual, considering their unique symptoms and constitution. No two patients are treated exactly alike.

## Vital Force
Homeopathy recognizes a vital force or energy that maintains health. Disease is seen as a disturbance of this force, and remedies help restore balance.

## Potentization
Substances are diluted and succussed (shaken) to enhance their therapeutic effect while minimizing toxicity. This process is believed to release the "spirit-like" healing power.

## Constitutional Types
- **Sanguine**: Cheerful, optimistic, warm, responsive
- **Choleric**: Ambitious, irritable, decisive, passionate
- **Melancholic**: Reflective, cautious, anxious, sensitive
- **Phlegmatic**: Calm, relaxed, sluggish, patient

# Remedy Indications by System

## Respiratory System
**Cough:**
- Aconite: Dry, hard, croupy cough
- Belladonna: Spasmodic, dry cough
- Bryonia: Dry, painful cough
- Hepar Sulph: Racking cough with choking

**Asthma:**
- Arsenicum: Anxious, worse at night
- Natrum Sulph: Humid, worse in damp weather
- Sambucus: Sudden attacks at night

**Bronchitis:**
- Hepar Sulph: Racking cough with choking
- Sulphur: Chronic, with discharge
- Antimonium Tart: Rattling cough with weakness

## Digestive System
**Constipation:**
- Nux Vomica: With ineffectual urging
- Alumina: From inactivity
- Bryonia: From dryness

**Diarrhea:**
- Arsenicum: From food poisoning
- Podophyllum: Painless, profuse
- Sulphur: At 5 AM, with urgency

**Acid Reflux:**
- Natrum Phos: Sour belching
- Robinia: Sour vomiting
- Nux Vomica: Bitter taste

## Nervous System
**Anxiety:**
- Aconite: With restlessness
- Arsenicum: With fear
- Gelsemium: With trembling

**Depression:**
- Sepia: With indifference
- Natrum Mur: With grief
- Aurum: With suicidal thoughts

**Insomnia:**
- Coffea: From excited thoughts
- Nux Vomica: From mental exertion
- Belladonna: With dreams

## Skin Conditions
**Eczema:**
- Sulphur: Itchy, burning
- Graphites: Weeping, sticky
- Mezereum: Thick crusts

**Acne:**
- Hepar Sulph: Pustular, painful
- Pulsatilla: From rich food
- Silicea: Keloids, scarring

**Psoriasis:**
- Arsenicum: Dry, scaly
- Sulphur: Exfoliating
- Petroleum: Cracked, bleeding

## Reproductive System
**Menstrual Issues:**
- Pulsatilla: Irregular, changeable
- Sepia: Bearing down, irritable
- Belladonna: Heavy, painful

**Menopause:**
- Sepia: Hot flashes, irritability
- Lachesis: Suffocative sensations
- Sulphur: Burning, sweating

## Male Issues:
- Nux Vomica: Premature ejaculation
- Lycopodium: Impotence
- Selenium: Debility

# Case Taking Guidelines

## Important Questions to Ask
1. What are your main symptoms?
2. When did they start?
3. What makes them better or worse?
4. What is your emotional state?
5. How do you sleep?
6. What are your food cravings/aversions?
7. What is your energy level?
8. Any history of trauma or grief?

## Constitutional Indicators
- Body type and build
- Temperature preferences
- Sleep position
- Food preferences
- Emotional patterns
- Response to weather

"""

        with open(self.data_path, 'w', encoding='utf-8') as f:
            f.write(sample_data)

        logger.info(
            f"✅ Created comprehensive sample homeopathy data at {self.data_path}")

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """Create and return the embedding model"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_vectorstore(self, documents: List) -> FAISS:
        """Create FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents to create vectorstore")

        # Try loading existing index
        if os.path.exists(self.index_path):
            try:
                logger.info("Loading existing vectorstore...")
                vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ Vectorstore loaded from disk")
                return vectorstore
            except Exception as e:
                logger.warning(f"Error loading existing vectorstore: {str(e)}")
                logger.info("Creating new vectorstore...")

        # Create new vectorstore
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)

            # Save for future use
            try:
                vectorstore.save_local(self.index_path)
                logger.info("✓ Vectorstore saved to disk")
            except Exception as e:
                logger.warning(f"Could not save vectorstore: {str(e)}")

            return vectorstore

        except Exception as e:
            logger.error(f"Error creating vectorstore: {str(e)}")
            raise

    def initialize_llm(self):
        """Initialize the LLM with proper configuration"""
        try:
            # Use TinyLlama (works well on CPU)
            model_to_use = self.model_name

            # Setup quantization for GPU
            if self.device == "cuda" and self.use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = None

            # Load tokenizer and model
            logger.info(f"Loading model: {model_to_use}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_to_use,
                trust_remote_code=True
            )

            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            if bnb_config:
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_use,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_use,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )

            # Create generation config
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                max_new_tokens=256,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                max_new_tokens=256,
                return_full_text=False,
                device_map="auto" if self.device == "cuda" else None
            )

            # Create HuggingFacePipeline LLM
            llm = HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 256,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.1
                }
            )

            logger.info(f"✅ LLM initialized: {model_to_use}")
            return llm

        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            logger.info("Using fallback LLM...")
            return self.create_fallback_llm()

    def create_fallback_llm(self):
        """Create a fallback LLM when the main model fails"""
        from langchain_community.llms import FakeLLM

        class HomeoFallbackLLM(FakeLLM):
            def _call(self, prompt: str, **kwargs) -> str:
                # Extract question from prompt
                question = prompt.split(
                    "Question:")[-1].split("Helpful Answer:")[0].strip()

                # Check if it's a greeting
                greetings = ["hi", "hello", "good morning",
                             "good afternoon", "good evening", "hey", "how are you"]
                if any(greeting in question.lower() for greeting in greetings):
                    return f"""Good morning! 🌞 

I'm your HomeoAI Assistant, here to help you with any questions about homeopathy. 

How can I assist you today? Whether you have questions about specific remedies, want to understand homeopathic principles, or need help analyzing symptoms, I'm here to help!

Please feel free to ask me anything about homeopathy. 😊
."""

                return f"""Based on your question about "{question}", here's general homeopathic guidance:

Homeopathy follows the principle of "like cures like." For proper treatment, consider:

1. **Constitutional Type**: Your overall physical and emotional makeup
2. **Modalities**: What makes symptoms better or worse
3. **Specific Symptoms**: Individual characteristics of your condition

Common approaches might include:
- For acute conditions: Consider Aconite, Belladonna, or Arsenicum
- For chronic conditions: Consider Sulphur, Lycopodium, or Nux Vomica
- For emotional symptoms: Consider Ignatia, Pulsatilla, or Natrum Mur

"""

        return HomeoFallbackLLM()

    def create_qa_chain(self):
        """Create the QA chain with proper prompt template"""
        template = """{system_prompt}

Use the following context to answer the question. If you don't know, say you don't know.

Context: {context}

Question: {question}

Helpful Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "system_prompt"]
        )

        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=False,
                return_source_documents=True,
                response_if_no_docs_found="I couldn't find specific information about that in my knowledge base. Could you please provide more details or rephrase your question?"
            )

            return chain

        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise

    def generate_response(self, query: str, history: List[Dict] = None) -> str:
        """Generate response using the QA chain"""
        try:
            if not query or not query.strip():
                return "Please provide a question or symptoms to analyze."

            # Check if it's a greeting or casual conversation
            greetings = ["hi", "hello", "good morning", "good afternoon",
                         "good evening", "hey", "how are you", "what's up"]
            query_lower = query.lower().strip()

            if any(greeting in query_lower for greeting in greetings):
                return self.handle_greeting(query)

            # Update memory with history if provided
            if history is not None:
                self.memory.clear()
                for item in history:
                    if item.get('role') == 'user':
                        self.memory.chat_memory.add_user_message(
                            item.get('content', ''))
                    elif item.get('role') == 'assistant':
                        self.memory.chat_memory.add_ai_message(
                            item.get('content', ''))

            # Get response from chain
            logger.info(f"📝 Processing query: {query[:100]}...")

            # Get relevant documents
            if self.retriever:
                docs = self.retriever.get_relevant_documents(query)
                if not docs:
                    logger.warning("No relevant documents found")
                    return self.handle_no_context(query)

            # Try to get response from QA chain
            try:
                result = self.qa_chain({"question": query})
                response = result.get(
                    'answer', "I couldn't generate a response. Please try again.")
            except Exception as e:
                logger.error(f"Error in QA chain: {str(e)}")
                return self.handle_error(query)

            # Add disclaimer if not present
            # disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only. Always consult a qualified homeopath or healthcare provider for proper diagnosis and treatment."

            # if disclaimer not in response:
            #     response += disclaimer

            # Log source documents if available
            if 'source_documents' in result:
                sources = result['source_documents']
                if sources:
                    logger.info(
                        f"Response based on {len(sources)} source documents")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self.handle_error(query)

    def handle_greeting(self, query: str) -> str:
        """Handle greeting messages with a warm response"""
        time_greeting = "Hello"
        if "morning" in query.lower():
            time_greeting = "Good morning! 🌞"
        elif "afternoon" in query.lower():
            time_greeting = "Good afternoon! 🌤️"
        elif "evening" in query.lower():
            time_greeting = "Good evening! 🌙"
        elif "night" in query.lower():
            time_greeting = "Good night! 🌙"

        return f"""{time_greeting} 

I'm your HomeoAI Assistant, here to help you with any questions about homeopathy. 

I can help you with:
- 💊 Information about specific remedies
- 📚 Understanding homeopathic principles
- 🤔 Analyzing symptoms (for educational purposes)
- 🔍 Finding remedies for specific conditions
- 📖 Learning about materia medica

How can I assist you today? 😊
"""

    def handle_no_context(self, query: str) -> str:
        """Handle cases where no relevant documents are found"""
        return f"""I appreciate your question about "{query}".

While I couldn't find specific information about this in my knowledge base, I can provide some general guidance:

1. **Be specific**: The more details you provide about your symptoms, the better I can help
2. **Consider modalities**: What makes your symptoms better or worse?
3. **Look at the whole picture**: Homeopathy considers physical, mental, and emotional aspects

You might want to:
- Rephrase your question with more details
- Ask about specific remedies you're interested in
- Describe your symptoms in more detail

I'm here to help! Let me know if you have more specific questions.

"""

    def handle_error(self, query: str) -> str:
        """Handle errors gracefully"""
        return f"""I apologize, but I encountered an issue processing your request.

For your question: "{query}"

I recommend:
1. Trying to rephrase your question more specifically
2. Providing more details about your symptoms
3. Asking about specific remedies you're interested in

I'm here to help with homeopathic information, so please feel free to ask again!

"""

    def search_remedies(self, query: str, k: int = 5) -> List[Dict]:
        """Search for remedies in the knowledge base"""
        try:
            if not self.retriever:
                logger.error("Retriever not initialized")
                return []

            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)

            # Format results
            results = []
            for doc in docs[:k]:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Error searching remedies: {str(e)}")
            return []

    def get_vectorstore_info(self) -> Dict:
        """Get information about the vector store"""
        try:
            if not self.vectorstore:
                return {"status": "Not initialized"}

            return {
                "status": "Initialized",
                "index_path": self.index_path,
                "embedding_model": self.embedding_model,
                "device": self.device,
                "k_retrieval": self.k_retrieval
            }

        except Exception as e:
            logger.error(f"Error getting vectorstore info: {str(e)}")
            return {"status": "Error", "error": str(e)}
