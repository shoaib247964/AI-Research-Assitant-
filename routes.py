import os
import uuid
import json
from datetime import datetime
from flask import render_template, request, jsonify, session, current_app
from werkzeug.utils import secure_filename
from app import app, db
from models import Document, Conversation
from langchain_service import LangChainService
import logging

# Initialize LangChain service
langchain_service = LangChainService()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page route"""
    # Ensure session has an ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Get uploaded documents
    documents = Document.query.order_by(Document.upload_time.desc()).all()
    
    # Get conversation history for this session
    conversations = Conversation.query.filter_by(
        session_id=session['session_id']
    ).order_by(Conversation.timestamp.asc()).all()
    
    return render_template('index.html', 
                         documents=documents, 
                         conversations=conversations)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        logging.info(f"Upload request received. Files: {list(request.files.keys())}")
        if 'file' not in request.files:
            logging.warning("No 'file' key in request.files")
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported. Please upload PDF or TXT files.'}), 400
        
        # Generate secure filename
        original_filename = file.filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Get file type
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Create database record
        document = Document(
            filename=unique_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_type=file_type
        )
        db.session.add(document)
        db.session.commit()
        
        # Process document with LangChain
        try:
            success, summary = langchain_service.process_document(file_path, document.id)
            if success:
                document.processed = True
                document.summary = summary
                db.session.commit()
                logging.info(f"Document {document.id} processed successfully")
            else:
                logging.error(f"Failed to process document {document.id}: {summary}")
        except Exception as e:
            logging.error(f"Error processing document: {e}")
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'document': document.to_dict()
        }), 200
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question asking"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        document_id = data.get('document_id')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']
        
        # Get conversation history for context
        conversation_history = Conversation.query.filter_by(
            session_id=session_id
        ).order_by(Conversation.timestamp.desc()).limit(10).all()
        
        # Ask question using LangChain service
        answer, context_used = langchain_service.ask_question(
            question, 
            document_id, 
            conversation_history
        )
        
        # Save conversation
        conversation = Conversation(
            session_id=session_id,
            document_id=document_id,
            question=question,
            answer=answer,
            context_used=context_used
        )
        db.session.add(conversation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'answer': answer,
            'conversation': conversation.to_dict()
        })
        
    except Exception as e:
        logging.error(f"Question asking error: {e}")
        return jsonify({'error': f'Failed to process question: {str(e)}'}), 500

@app.route('/documents')
def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = Document.query.order_by(Document.upload_time.desc()).all()
        return jsonify({
            'success': True,
            'documents': [doc.to_dict() for doc in documents]
        })
    except Exception as e:
        logging.error(f"Error getting documents: {e}")
        return jsonify({'error': 'Failed to get documents'}), 500

@app.route('/conversations')
def get_conversations():
    """Get conversation history for current session"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': True, 'conversations': []})
        
        conversations = Conversation.query.filter_by(
            session_id=session_id
        ).order_by(Conversation.timestamp.asc()).all()
        
        return jsonify({
            'success': True,
            'conversations': [conv.to_dict() for conv in conversations]
        })
    except Exception as e:
        logging.error(f"Error getting conversations: {e}")
        return jsonify({'error': 'Failed to get conversations'}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear current session conversation history"""
    try:
        session_id = session.get('session_id')
        if session_id:
            # Delete conversations for this session
            Conversation.query.filter_by(session_id=session_id).delete()
            db.session.commit()
            
            # Generate new session ID
            session['session_id'] = str(uuid.uuid4())
        
        return jsonify({'success': True, 'message': 'Session cleared'})
    except Exception as e:
        logging.error(f"Error clearing session: {e}")
        return jsonify({'error': 'Failed to clear session'}), 500

@app.route('/delete_document/<int:document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a document and its associated data"""
    try:
        document = Document.query.get_or_404(document_id)
        
        # Delete file from filesystem
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database
        Conversation.query.filter_by(document_id=document_id).delete()
        db.session.delete(document)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Document deleted successfully'})
    except Exception as e:
        logging.error(f"Error deleting document: {e}")
        return jsonify({'error': 'Failed to delete document'}), 500

@app.route('/compare_documents', methods=['POST'])
def compare_documents():
    """Compare two or more documents"""
    try:
        data = request.get_json()
        document_ids = data.get('document_ids', [])
        comparison_type = data.get('type', 'similarities')  # 'similarities', 'differences', 'themes'
        
        if len(document_ids) < 2:
            return jsonify({'error': 'At least 2 documents required for comparison'}), 400
        
        # Get documents
        documents = Document.query.filter(Document.id.in_(document_ids)).all()
        if len(documents) != len(document_ids):
            return jsonify({'error': 'One or more documents not found'}), 404
        
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']
        
        # Perform comparison
        result = langchain_service.compare_documents(documents, comparison_type)
        
        # Save comparison as conversation
        question = f"Compare documents: {', '.join([doc.original_filename for doc in documents])}"
        conversation = Conversation(
            session_id=session_id,
            document_id=None,  # Multi-document comparison
            question=question,
            answer=result,
            context_used=f"Compared {len(documents)} documents"
        )
        db.session.add(conversation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'comparison': result,
            'conversation': conversation.to_dict()
        })
        
    except Exception as e:
        logging.error(f"Error comparing documents: {e}")
        return jsonify({'error': f'Failed to compare documents: {str(e)}'}), 500

@app.route('/export_conversation', methods=['POST'])
def export_conversation():
    """Export conversation history as text"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        conversations = Conversation.query.filter_by(
            session_id=session_id
        ).order_by(Conversation.timestamp.asc()).all()
        
        if not conversations:
            return jsonify({'error': 'No conversations to export'}), 400
        
        # Generate export text
        export_text = "# AI Research Assistant Conversation Export\n\n"
        export_text += f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        export_text += f"Session ID: {session_id}\n\n"
        export_text += "---\n\n"
        
        for conv in conversations:
            export_text += f"**Q:** {conv.question}\n\n"
            export_text += f"**A:** {conv.answer}\n\n"
            export_text += f"*Time: {conv.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            export_text += "---\n\n"
        
        return jsonify({
            'success': True,
            'export_text': export_text,
            'filename': f"research_conversation_{session_id[:8]}.md"
        })
        
    except Exception as e:
        logging.error(f"Error exporting conversation: {e}")
        return jsonify({'error': 'Failed to export conversation'}), 500

@app.route('/search_documents', methods=['POST'])
def search_documents():
    """Search across all documents using semantic search"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']
        
        # Perform semantic search
        results = langchain_service.search_documents(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Error searching documents: {e}")
        return jsonify({'error': f'Failed to search documents: {str(e)}'}), 500
