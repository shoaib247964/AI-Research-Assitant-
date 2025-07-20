from app import db
from datetime import datetime
import json

class Document(db.Model):
    """Model for storing uploaded documents"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    summary = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_type': self.file_type,
            'upload_time': self.upload_time.isoformat(),
            'processed': self.processed,
            'summary': self.summary
        }

class Conversation(db.Model):
    """Model for storing conversation history"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    context_used = db.Column(db.Text)  # Store relevant document chunks
    
    document = db.relationship('Document', backref='conversations')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'document_id': self.document_id,
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp.isoformat(),
            'context_used': self.context_used
        }
