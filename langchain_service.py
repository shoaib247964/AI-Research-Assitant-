import os
import logging
from typing import List, Tuple, Optional
import faiss
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from openai import OpenAI

class LangChainService:
    """Service class for handling LangChain operations"""
    
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o",
            openai_api_key=self.openai_api_key
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Store for document vectors
        self.vector_stores = {}  # document_id -> FAISS vector store
        self.document_texts = {}  # document_id -> original text
        
        # Memory for conversations
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        logging.info("LangChain service initialized")
    
    def process_document(self, file_path: str, document_id: int) -> Tuple[bool, str]:
        """Process a document and create vector embeddings"""
        try:
            # Load document based on file type
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                return False, "Unsupported file type"
            
            # Load and split document
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            
            if not texts:
                return False, "No text content found in document"
            
            # Store original text for reference
            full_text = "\n".join([doc.page_content for doc in documents])
            self.document_texts[document_id] = full_text
            
            # Create vector store
            vector_store = FAISS.from_documents(texts, self.embeddings)
            self.vector_stores[document_id] = vector_store
            
            # Generate summary
            summary = self._generate_summary(full_text)
            
            logging.info(f"Successfully processed document {document_id}")
            return True, summary
            
        except Exception as e:
            logging.error(f"Error processing document {document_id}: {e}")
            return False, f"Processing failed: {str(e)}"
    
    def _generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the document using OpenAI"""
        try:
            # Truncate text if too long for API
            if len(text) > 3000:
                text = text[:3000] + "..."
            
            prompt = f"""Please provide a concise summary of the following document in about 2-3 sentences. 
            Focus on the main topics, key points, and overall purpose of the document.
            
            Document text:
            {text}
            
            Summary:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return "Summary generation failed"
    
    def ask_question(self, question: str, document_id: Optional[int] = None, 
                    conversation_history: List = None) -> Tuple[str, str]:
        """Ask a question about documents or general conversation"""
        try:
            if document_id and document_id in self.vector_stores:
                # Question about specific document
                return self._ask_document_question(question, document_id, conversation_history)
            elif self.vector_stores:
                # Question about all documents
                return self._ask_general_question(question, conversation_history)
            else:
                # No documents available, general conversation
                return self._ask_general_conversation(question, conversation_history)
                
        except Exception as e:
            logging.error(f"Error asking question: {e}")
            return f"Sorry, I encountered an error: {str(e)}", ""
    
    def _ask_document_question(self, question: str, document_id: int, 
                              conversation_history: List = None) -> Tuple[str, str]:
        """Ask question about a specific document"""
        vector_store = self.vector_stores[document_id]
        
        # Create retrieval chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Build conversation history for context
        chat_history = []
        if conversation_history:
            for conv in reversed(conversation_history[:5]):  # Last 5 conversations
                chat_history.append((conv.question, conv.answer))
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        # Ask question
        result = qa_chain({
            "question": question,
            "chat_history": chat_history
        })
        
        # Extract context from source documents
        context_used = ""
        if result.get("source_documents"):
            context_used = "\n".join([doc.page_content[:200] + "..." 
                                    for doc in result["source_documents"]])
        
        return result["answer"], context_used
    
    def _ask_general_question(self, question: str, conversation_history: List = None) -> Tuple[str, str]:
        """Ask question across all available documents"""
        if not self.vector_stores:
            return self._ask_general_conversation(question, conversation_history)
        
        # Combine all vector stores for search
        all_docs = []
        for doc_id, vector_store in self.vector_stores.items():
            # Get relevant docs from each vector store
            docs = vector_store.similarity_search(question, k=2)
            all_docs.extend(docs)
        
        if not all_docs:
            return self._ask_general_conversation(question, conversation_history)
        
        # Create context from all relevant documents
        context = "\n".join([doc.page_content for doc in all_docs[:5]])
        
        # Build conversation history
        history_context = ""
        if conversation_history:
            for conv in reversed(conversation_history[:3]):
                history_context += f"Q: {conv.question}\nA: {conv.answer}\n\n"
        
        # Create prompt with context
        prompt = f"""Based on the following document context and conversation history, please answer the question.
        
        Previous conversation:
        {history_context}
        
        Document context:
        {context}
        
        Question: {question}
        
        Please provide a helpful and accurate answer based on the available information."""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content, context[:500]
    
    def _ask_general_conversation(self, question: str, conversation_history: List = None) -> Tuple[str, str]:
        """Handle general conversation without document context"""
        # Build conversation history
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. You can answer questions and have conversations, but you work best when provided with documents to analyze."}
        ]
        
        # Add conversation history
        if conversation_history:
            for conv in reversed(conversation_history[-5:]):  # Last 5 conversations
                messages.append({"role": "user", "content": conv.question})
                messages.append({"role": "assistant", "content": conv.answer})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content, ""
    
    def compare_documents(self, documents: List, comparison_type: str = 'similarities') -> str:
        """Compare multiple documents"""
        try:
            document_summaries = []
            document_contents = []
            
            for doc in documents:
                if doc.id in self.document_texts:
                    content = self.document_texts[doc.id][:2000]  # First 2000 chars
                    document_contents.append(f"Document: {doc.original_filename}\n{content}")
                    document_summaries.append(f"- {doc.original_filename}: {doc.summary}")
            
            if comparison_type == 'similarities':
                prompt_instruction = "identify key similarities, common themes, and shared concepts between"
            elif comparison_type == 'differences':
                prompt_instruction = "identify key differences, contrasting viewpoints, and unique aspects of"
            else:  # themes
                prompt_instruction = "extract and analyze the main themes across"
            
            prompt = f"""Please {prompt_instruction} the following documents. Provide a detailed analysis with specific examples.

Document Summaries:
{chr(10).join(document_summaries)}

Full Content Analysis:
{chr(10).join(document_contents)}

Provide a comprehensive comparison covering:
1. Main points of comparison
2. Specific examples from each document
3. Key insights or conclusions"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error comparing documents: {e}")
            return f"Error comparing documents: {str(e)}"
    
    def search_documents(self, query: str) -> List[dict]:
        """Search across all documents using semantic similarity"""
        try:
            results = []
            
            for doc_id, vector_store in self.vector_stores.items():
                # Get document info
                from models import Document
                document = Document.query.get(doc_id)
                if not document:
                    continue
                
                # Search in this document
                similar_docs = vector_store.similarity_search_with_score(query, k=3)
                
                for doc, score in similar_docs:
                    results.append({
                        'document_id': doc_id,
                        'document_name': document.original_filename,
                        'content': doc.page_content[:300] + "...",
                        'similarity_score': float(score),
                        'relevance': 'High' if score < 0.5 else 'Medium' if score < 1.0 else 'Low'
                    })
            
            # Sort by similarity score (lower is better for FAISS)
            results.sort(key=lambda x: x['similarity_score'])
            
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            return []
