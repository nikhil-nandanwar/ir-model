"""
Document indexing and storage for the information retrieval system.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    TRANSFORMERS_AVAILABLE = False

if not SKLEARN_AVAILABLE:
    print("Warning: scikit-learn not installed. TF-IDF features will be limited.")
if not TRANSFORMERS_AVAILABLE:
    print("Warning: sentence-transformers not installed. BERT features disabled.")

from .utils import TextPreprocessor, SimilarityCalculator


class Document:
    """Represents a document in the IR system."""
    
    def __init__(self, doc_id: str, title: str, content: str, 
                 categories: List[str] = None, metadata: Dict = None):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.categories = categories or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.processed_content = ""
        self.keywords = []
        
    def to_dict(self) -> Dict:
        """Convert document to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content,
            'categories': self.categories,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'processed_content': self.processed_content,
            'keywords': self.keywords
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create document from dictionary."""
        doc = cls(
            doc_id=data['doc_id'],
            title=data['title'],
            content=data['content'],
            categories=data.get('categories', []),
            metadata=data.get('metadata', {})
        )
        if 'created_at' in data:
            doc.created_at = datetime.fromisoformat(data['created_at'])
        doc.processed_content = data.get('processed_content', '')
        doc.keywords = data.get('keywords', [])
        return doc


class DocumentIndex:
    """Manages document storage, indexing, and retrieval."""
    
    def __init__(self, use_bert: bool = True):
        self.documents: Dict[str, Document] = {}
        self.preprocessor = TextPreprocessor()
        self.similarity_calc = SimilarityCalculator()
        
        # TF-IDF components
        if SKLEARN_AVAILABLE and TfidfVectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # BERT embeddings
        self.use_bert = use_bert and TRANSFORMERS_AVAILABLE
        self.bert_model = None
        self.bert_embeddings = {}
        
        if self.use_bert and SentenceTransformer:
            try:
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.use_bert = False
        
        # Inverted index for fast keyword lookup
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        
        # Category index
        self.category_index: Dict[str, List[str]] = defaultdict(list)
    
    def add_document(self, doc_id: str, title: str, content: str,
                    categories: List[str] = None, metadata: Dict = None) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            content: Document content
            categories: List of categories/tags
            metadata: Additional metadata
        """
        # Create document object
        doc = Document(doc_id, title, content, categories, metadata)
        
        # Preprocess content
        full_text = f"{title} {content}"
        doc.processed_content = self.preprocessor.preprocess_text(full_text)
        doc.keywords = self.preprocessor.extract_keywords(full_text)
        
        # Store document
        self.documents[doc_id] = doc
        
        # Update inverted index
        for keyword in doc.keywords:
            self.inverted_index[keyword].append(doc_id)
        
        # Update category index
        for category in doc.categories:
            self.category_index[category].append(doc_id)
        
        # Update TF-IDF matrix
        self._update_tfidf_matrix()
        
        # Update BERT embeddings
        if self.use_bert and self.bert_model:
            self._update_bert_embeddings(doc_id, full_text)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.
        
        Args:
            doc_id: Document identifier to remove
            
        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        
        # Remove from inverted index
        for keyword in doc.keywords:
            if doc_id in self.inverted_index[keyword]:
                self.inverted_index[keyword].remove(doc_id)
                if not self.inverted_index[keyword]:
                    del self.inverted_index[keyword]
        
        # Remove from category index
        for category in doc.categories:
            if doc_id in self.category_index[category]:
                self.category_index[category].remove(doc_id)
                if not self.category_index[category]:
                    del self.category_index[category]
        
        # Remove from BERT embeddings
        if doc_id in self.bert_embeddings:
            del self.bert_embeddings[doc_id]
        
        # Remove document
        del self.documents[doc_id]
        
        # Update TF-IDF matrix
        self._update_tfidf_matrix()
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def search_keywords(self, keywords: List[str]) -> List[str]:
        """
        Search documents by keywords using inverted index.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of document IDs matching keywords
        """
        if not keywords:
            return []
        
        # Find documents containing any of the keywords
        candidate_docs = set()
        for keyword in keywords:
            if keyword in self.inverted_index:
                candidate_docs.update(self.inverted_index[keyword])
        
        return list(candidate_docs)
    
    def search_categories(self, categories: List[str]) -> List[str]:
        """
        Search documents by categories.
        
        Args:
            categories: List of categories to search for
            
        Returns:
            List of document IDs in specified categories
        """
        if not categories:
            return []
        
        candidate_docs = set()
        for category in categories:
            if category in self.category_index:
                candidate_docs.update(self.category_index[category])
        
        return list(candidate_docs)
    
    def calculate_tfidf_similarity(self, query: str, doc_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate TF-IDF similarity scores for query against documents.
        
        Args:
            query: Search query
            doc_ids: Specific document IDs to compare (None for all)
            
        Returns:
            Dictionary mapping doc_id to similarity score
        """
        if not SKLEARN_AVAILABLE or self.tfidf_matrix is None or self.tfidf_vectorizer is None:
            # Fallback to simple keyword matching
            return self._simple_keyword_similarity(query, doc_ids)
        
        # Process query
        processed_query = self.preprocessor.preprocess_text(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Map to document IDs
        doc_list = list(self.documents.keys())
        scores = {}
        
        for i, doc_id in enumerate(doc_list):
            if doc_ids is None or doc_id in doc_ids:
                scores[doc_id] = similarities[i]
        
        return scores
    
    def _simple_keyword_similarity(self, query: str, doc_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Simple keyword-based similarity fallback when TF-IDF is not available.
        """
        processed_query = self.preprocessor.preprocess_text(query)
        query_words = set(processed_query.split())
        
        scores = {}
        target_docs = doc_ids if doc_ids else list(self.documents.keys())
        
        for doc_id in target_docs:
            doc = self.documents.get(doc_id)
            if doc:
                doc_words = set(doc.processed_content.split())
                if len(query_words) > 0 and len(doc_words) > 0:
                    # Simple Jaccard similarity
                    intersection = len(query_words.intersection(doc_words))
                    union = len(query_words.union(doc_words))
                    scores[doc_id] = intersection / union if union > 0 else 0.0
                else:
                    scores[doc_id] = 0.0
            else:
                scores[doc_id] = 0.0
        
        return scores
    
    def calculate_bert_similarity(self, query: str, doc_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate BERT similarity scores for query against documents.
        
        Args:
            query: Search query
            doc_ids: Specific document IDs to compare (None for all)
            
        Returns:
            Dictionary mapping doc_id to similarity score
        """
        if not self.use_bert or not self.bert_model:
            return {}
        
        # Get query embedding
        query_embedding = self.bert_model.encode([query])[0]
        
        scores = {}
        target_docs = doc_ids if doc_ids else list(self.documents.keys())
        
        for doc_id in target_docs:
            if doc_id in self.bert_embeddings:
                doc_embedding = self.bert_embeddings[doc_id]
                similarity = self.similarity_calc.cosine_similarity(
                    query_embedding, doc_embedding
                )
                scores[doc_id] = similarity
        
        return scores
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories in the index."""
        return list(self.category_index.keys())
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    def get_documents_by_category(self, category: str) -> List[Document]:
        """Get all documents in a specific category."""
        doc_ids = self.category_index.get(category, [])
        return [self.documents[doc_id] for doc_id in doc_ids]
    
    def _update_tfidf_matrix(self) -> None:
        """Update the TF-IDF matrix with all documents."""
        if not self.documents or not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            self.tfidf_matrix = None
            return
        
        # Prepare document texts
        doc_texts = []
        for doc in self.documents.values():
            doc_texts.append(doc.processed_content)
        
        # Fit and transform
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
        except Exception as e:
            print(f"Warning: TF-IDF update failed: {e}")
            self.tfidf_matrix = None
    
    def _update_bert_embeddings(self, doc_id: str, text: str) -> None:
        """Update BERT embeddings for a document."""
        if not self.use_bert or not self.bert_model:
            return
        
        try:
            embedding = self.bert_model.encode([text])[0]
            self.bert_embeddings[doc_id] = embedding
        except Exception as e:
            print(f"Warning: BERT embedding failed for {doc_id}: {e}")
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save document index to file.
        
        Args:
            filepath: Path to save the index
        """
        data = {
            'documents': {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            'inverted_index': dict(self.inverted_index),
            'category_index': dict(self.category_index),
            'use_bert': self.use_bert
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load document index from file.
        
        Args:
            filepath: Path to load the index from
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load documents
        self.documents = {}
        for doc_id, doc_data in data.get('documents', {}).items():
            self.documents[doc_id] = Document.from_dict(doc_data)
        
        # Load indexes
        self.inverted_index = defaultdict(list, data.get('inverted_index', {}))
        self.category_index = defaultdict(list, data.get('category_index', {}))
        
        # Rebuild TF-IDF matrix and BERT embeddings
        self._update_tfidf_matrix()
        if self.use_bert and self.bert_model:
            for doc_id, doc in self.documents.items():
                full_text = f"{doc.title} {doc.content}"
                self._update_bert_embeddings(doc_id, full_text)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_documents': len(self.documents),
            'total_categories': len(self.category_index),
            'total_keywords': len(self.inverted_index),
            'categories': list(self.category_index.keys()),
            'use_bert': self.use_bert,
            'has_tfidf': self.tfidf_matrix is not None
        }