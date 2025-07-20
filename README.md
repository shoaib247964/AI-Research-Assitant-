AI-Powered Research Assistant Agent 
## Overview

This is a Flask-based AI research assistant that allows users to upload documents (PDF and TXT), process them using LangChain and Perplexity API , and engage in conversational Q&A about the document content. The application uses vector embeddings for semantic search and maintains conversation history with memory capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask web application with SQLAlchemy ORM
- **Database**: SQLite (default) with support for PostgreSQL via environment variables
- **AI/ML Stack**: LangChain + OpenAI GPT-4o for document processing and conversational AI
- **Vector Storage**: FAISS for storing document embeddings and semantic search
- **File Processing**: Support for PDF and TXT document uploads with chunking

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap 5 dark theme
- **Styling**: Bootstrap CSS framework with custom CSS for chat interface and advanced UI
- **JavaScript**: Vanilla JavaScript with enhanced features for document management and analysis
- **UI Components**: 
  - Sidebar for document management with checkboxes and dropdown menus
  - Main chat area for conversations with typing indicators
  - Advanced features panel with search, comparison, and export tools
  - Modal dialogs for comparison type selection
  - Responsive design with mobile support

## Key Components

### Models (`models.py`)
- **Document Model**: Stores uploaded file metadata, processing status, and summaries
- **Conversation Model**: Tracks Q&A history with session management and context storage

### LangChain Service (`langchain_service.py`)
- **Document Processing**: Loads and chunks PDF/TXT files using RecursiveCharacterTextSplitter
- **Embeddings**: OpenAI embeddings for semantic vector representation
- **Vector Search**: FAISS vector store for similarity search
- **Conversational Chain**: ConversationalRetrievalChain with memory for context-aware responses
- **Memory Management**: ConversationBufferMemory for maintaining chat history
- **Document Comparison**: Advanced AI-powered document comparison with multiple analysis types
- **Semantic Search**: Cross-document search using vector embeddings

### Routes (`routes.py`)
- **File Upload**: Handles document upload, validation, and processing
- **Chat Interface**: Processes user questions and returns AI responses
- **Session Management**: Maintains user sessions and conversation history
- **Document Management**: Delete documents with cleanup of associated data
- **Document Comparison**: Compare 2+ documents for similarities, differences, or themes
- **Conversation Export**: Export chat history to markdown format
- **Semantic Search**: Search across all documents using AI-powered similarity matching

## Data Flow

1. **Document Upload**: User uploads PDF/TXT → File validation → Storage in uploads folder → Database record creation
2. **Document Processing**: LangChain loads document → Text chunking → Embedding generation → FAISS vector store creation → Auto-summary generation
3. **Question Processing**: User question → Semantic search in vector store → Context retrieval → LLM processing → Response generation
4. **Memory Management**: Conversation history stored in database and maintained in LangChain memory for context
5. **Document Comparison**: Multiple documents selected → Content analysis → AI comparison → Structured response with similarities/differences/themes
6. **Semantic Search**: Search query → Vector similarity across all documents → Ranked results with relevance scores
7. **Export**: Conversation history → Formatted markdown → Download as file

## External Dependencies

### Core Dependencies
- **Flask**: Web framework and routing
- **SQLAlchemy**: Database ORM and migrations
- **LangChain**: Document processing, embeddings, and conversational AI
- **OpenAI**: GPT-4o language model and embeddings API
- **FAISS**: Vector similarity search library

### Frontend Dependencies
- **Bootstrap 5**: UI framework with dark theme
- **Font Awesome**: Icon library
- **Vanilla JavaScript**: Frontend interactivity

### Environment Variables Required
- `OPENAI_API_KEY`: OpenAI API authentication
- `DATABASE_URL`: Database connection string (optional, defaults to SQLite)
- `SESSION_SECRET`: Flask session security key

## Deployment Strategy

### Development Setup
- **Entry Point**: `main.py` runs Flask development server on port 5000
- **Database**: Auto-creates SQLite database and tables on startup
- **File Storage**: Local uploads folder for document storage
- **Debug Mode**: Enabled for development with detailed logging

### Production Considerations
- Database migration to PostgreSQL recommended for scalability
- File storage should be moved to cloud storage (S3, etc.)
- Environment variables must be properly configured
- Vector store persistence needs implementation for production
- Session management should use Redis or similar for distributed deployments

### Configuration
- **File Limits**: 16MB maximum file size
- **Supported Formats**: PDF and TXT files only
- **LLM Model**: GPT-4o (latest OpenAI model as of May 2024)
- **Chunking Strategy**: 1000 character chunks with 200 character overlap
- **Memory Type**: Buffer memory for conversation history

## Recent Changes (July 20, 2025)

### Fixed Upload Issues
- ✓ Added proper HTTP status codes to upload responses
- ✓ Enhanced JavaScript error handling with debug logging
- ✓ Fixed LangChain import deprecation warnings using newer modules

### Advanced Features Added
- ✓ **Document Management**: Delete documents with full cleanup of files and database records
- ✓ **Document Comparison**: AI-powered comparison tool with three analysis types:
  - Find Similarities: Identifies common themes and shared concepts
  - Find Differences: Highlights contrasting viewpoints and unique aspects  
  - Extract Themes: Analyzes main themes across multiple documents
- ✓ **Semantic Search**: Cross-document search using vector embeddings with relevance scoring
- ✓ **Export Functionality**: Download conversation history as markdown files
- ✓ **Enhanced UI**: Added checkboxes for document selection, dropdown menus, advanced features panel
- ✓ **Modal Dialogs**: Interactive comparison type selection with Bootstrap modals

### Technical Improvements
- ✓ Updated to latest LangChain modules (langchain-openai, langchain-community)
- ✓ Enhanced error handling across all new features
- ✓ Added proper request validation and response formatting
- ✓ Improved document list UI with management controls
