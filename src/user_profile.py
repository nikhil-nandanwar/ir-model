"""
User profile and search history management for personalized information retrieval.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np

from .utils import TextPreprocessor, TemporalWeighting, get_topic_categories


class SearchHistoryEntry:
    """Represents a single search history entry."""
    
    def __init__(self, query: str, timestamp: datetime = None, 
                 clicked_docs: List[str] = None, categories: List[str] = None):
        self.query = query
        self.timestamp = timestamp or datetime.now()
        self.clicked_docs = clicked_docs or []
        self.categories = categories or []
        self.processed_query = ""
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'timestamp': self.timestamp.isoformat(),
            'clicked_docs': self.clicked_docs,
            'categories': self.categories,
            'processed_query': self.processed_query
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SearchHistoryEntry':
        """Create from dictionary."""
        entry = cls(
            query=data['query'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            clicked_docs=data.get('clicked_docs', []),
            categories=data.get('categories', [])
        )
        entry.processed_query = data.get('processed_query', '')
        return entry


class UserProfile:
    """Manages user search history and interest profile."""
    
    def __init__(self, user_id: str, max_history_size: int = 1000):
        self.user_id = user_id
        self.max_history_size = max_history_size
        
        # Search history
        self.search_history: List[SearchHistoryEntry] = []
        
        # Interest modeling
        self.interest_categories: Dict[str, float] = defaultdict(float)
        self.keyword_preferences: Dict[str, float] = defaultdict(float)
        self.topic_preferences: Dict[str, float] = defaultdict(float)
        
        # Temporal settings
        self.temporal_weighting = TemporalWeighting()
        self.history_decay_rate = 0.05  # Decay rate for exponential weighting
        self.recency_window_hours = 24  # Hours for recency boost
        
        # Preprocessing
        self.preprocessor = TextPreprocessor()
        self.topic_categories = get_topic_categories()
        
        # Feedback learning
        self.click_through_data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.relevance_feedback: Dict[str, float] = {}
    
    def add_search(self, query: str, clicked_docs: List[str] = None, 
                  categories: List[str] = None) -> None:
        """
        Add a search query to the user's history.
        
        Args:
            query: Search query string
            clicked_docs: List of document IDs that were clicked
            categories: Categories of the search (if known)
        """
        # Create history entry
        entry = SearchHistoryEntry(query, clicked_docs=clicked_docs, categories=categories)
        entry.processed_query = self.preprocessor.preprocess_text(query)
        
        # Add to history
        self.search_history.append(entry)
        
        # Maintain history size limit
        if len(self.search_history) > self.max_history_size:
            self.search_history = self.search_history[-self.max_history_size:]
        
        # Update interest profile
        self._update_interests(entry)
        
        # Update click-through data
        if clicked_docs:
            for doc_id in clicked_docs:
                self.click_through_data[query][doc_id] += 1
    
    def get_user_context_vector(self, current_time: datetime = None) -> Dict[str, float]:
        """
        Generate a context vector representing user's current interests.
        
        Args:
            current_time: Current timestamp for temporal weighting
            
        Returns:
            Dictionary with weighted interest scores
        """
        if not self.search_history:
            return {}
        
        current_time = current_time or datetime.now()
        context_vector = defaultdict(float)
        
        # Weight search history by recency
        for entry in self.search_history:
            # Calculate temporal weight
            time_diff = (current_time - entry.timestamp).total_seconds() / 3600
            temporal_weight = self.temporal_weighting.exponential_decay(
                time_diff, self.history_decay_rate
            )
            
            # Apply recency boost
            recency_boost = self.temporal_weighting.recency_boost(
                time_diff, self.recency_window_hours
            )
            
            final_weight = temporal_weight * recency_boost
            
            # Add query keywords with temporal weighting
            query_keywords = entry.processed_query.split()
            for keyword in query_keywords:
                context_vector[f'keyword_{keyword}'] += final_weight
            
            # Add categories with temporal weighting
            for category in entry.categories:
                context_vector[f'category_{category}'] += final_weight * 2  # Higher weight for categories
        
        return dict(context_vector)
    
    def get_interest_profile(self) -> Dict[str, Any]:
        """
        Get comprehensive interest profile.
        
        Returns:
            Dictionary containing various interest metrics
        """
        return {
            'categories': dict(self.interest_categories),
            'keywords': dict(self.keyword_preferences),
            'topics': dict(self.topic_preferences),
            'total_searches': len(self.search_history),
            'recent_searches': len([
                entry for entry in self.search_history
                if (datetime.now() - entry.timestamp).days <= 7
            ])
        }
    
    def get_query_expansion_terms(self, query: str, max_terms: int = 5) -> List[str]:
        """
        Get expansion terms for a query based on user history.
        
        Args:
            query: Original query
            max_terms: Maximum number of expansion terms
            
        Returns:
            List of expansion terms
        """
        processed_query = self.preprocessor.preprocess_text(query)
        query_words = set(processed_query.split())
        
        # Find related keywords from history
        expansion_candidates = defaultdict(float)
        
        for entry in self.search_history:
            # Skip if queries are too different
            entry_words = set(entry.processed_query.split())
            if not query_words.intersection(entry_words):
                continue
            
            # Calculate temporal weight
            time_diff = (datetime.now() - entry.timestamp).total_seconds() / 3600
            weight = self.temporal_weighting.exponential_decay(time_diff, 0.1)
            
            # Add related words from similar queries
            for word in entry_words - query_words:
                expansion_candidates[word] += weight
        
        # Sort by weight and return top terms
        sorted_terms = sorted(expansion_candidates.items(), key=lambda x: x[1], reverse=True)
        return [term for term, weight in sorted_terms[:max_terms]]
    
    def calculate_document_preference_score(self, doc_categories: List[str], 
                                          doc_keywords: List[str]) -> float:
        """
        Calculate preference score for a document based on user profile.
        
        Args:
            doc_categories: Document categories
            doc_keywords: Document keywords
            
        Returns:
            Preference score (0-1)
        """
        score = 0.0
        
        # Category matching
        for category in doc_categories:
            if category in self.interest_categories:
                score += self.interest_categories[category] * 0.4
        
        # Keyword matching
        for keyword in doc_keywords:
            if keyword in self.keyword_preferences:
                score += self.keyword_preferences[keyword] * 0.3
        
        # Topic matching
        for topic, keywords in self.topic_categories.items():
            topic_match = len(set(doc_keywords).intersection(set(keywords)))
            if topic_match > 0 and topic in self.topic_preferences:
                score += (topic_match / len(keywords)) * self.topic_preferences[topic] * 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_similar_queries(self, query: str, max_queries: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar queries from search history.
        
        Args:
            query: Input query
            max_queries: Maximum number of similar queries to return
            
        Returns:
            List of tuples (similar_query, similarity_score)
        """
        processed_query = self.preprocessor.preprocess_text(query)
        query_words = set(processed_query.split())
        
        similar_queries = []
        
        for entry in self.search_history:
            entry_words = set(entry.processed_query.split())
            
            # Calculate Jaccard similarity
            if len(query_words) == 0 or len(entry_words) == 0:
                continue
                
            intersection = len(query_words.intersection(entry_words))
            union = len(query_words.union(entry_words))
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Minimum similarity threshold
                    # Apply temporal weighting
                    time_diff = (datetime.now() - entry.timestamp).total_seconds() / 3600
                    temporal_weight = self.temporal_weighting.exponential_decay(time_diff, 0.1)
                    
                    final_score = similarity * temporal_weight
                    similar_queries.append((entry.query, final_score))
        
        # Sort by similarity and return top queries
        similar_queries.sort(key=lambda x: x[1], reverse=True)
        return similar_queries[:max_queries]
    
    def update_click_feedback(self, query: str, doc_id: str, clicked: bool) -> None:
        """
        Update click-through feedback for learning.
        
        Args:
            query: Search query
            doc_id: Document ID
            clicked: Whether the document was clicked
        """
        feedback_key = f"{query}_{doc_id}"
        
        if clicked:
            self.relevance_feedback[feedback_key] = self.relevance_feedback.get(feedback_key, 0) + 0.1
        else:
            self.relevance_feedback[feedback_key] = max(
                self.relevance_feedback.get(feedback_key, 0) - 0.05, -0.5
            )
    
    def get_relevance_score(self, query: str, doc_id: str) -> float:
        """
        Get learned relevance score for query-document pair.
        
        Args:
            query: Search query
            doc_id: Document ID
            
        Returns:
            Relevance score (-0.5 to positive)
        """
        feedback_key = f"{query}_{doc_id}"
        return self.relevance_feedback.get(feedback_key, 0.0)
    
    def _update_interests(self, entry: SearchHistoryEntry) -> None:
        """Update interest profiles based on search history entry."""
        # Update keyword preferences
        query_keywords = entry.processed_query.split()
        for keyword in query_keywords:
            self.keyword_preferences[keyword] += 1.0
        
        # Update category preferences
        for category in entry.categories:
            self.interest_categories[category] += 2.0
        
        # Infer topic preferences from keywords
        for topic, topic_keywords in self.topic_categories.items():
            topic_match = len(set(query_keywords).intersection(set(topic_keywords)))
            if topic_match > 0:
                self.topic_preferences[topic] += topic_match
        
        # Apply decay to prevent unbounded growth
        self._apply_interest_decay()
    
    def _apply_interest_decay(self, decay_factor: float = 0.99) -> None:
        """Apply decay to interest scores to prevent unbounded growth."""
        for key in self.keyword_preferences:
            self.keyword_preferences[key] *= decay_factor
        
        for key in self.interest_categories:
            self.interest_categories[key] *= decay_factor
        
        for key in self.topic_preferences:
            self.topic_preferences[key] *= decay_factor
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save user profile to file.
        
        Args:
            filepath: Path to save the profile
        """
        data = {
            'user_id': self.user_id,
            'search_history': [entry.to_dict() for entry in self.search_history],
            'interest_categories': dict(self.interest_categories),
            'keyword_preferences': dict(self.keyword_preferences),
            'topic_preferences': dict(self.topic_preferences),
            'click_through_data': {k: dict(v) for k, v in self.click_through_data.items()},
            'relevance_feedback': dict(self.relevance_feedback),
            'max_history_size': self.max_history_size,
            'history_decay_rate': self.history_decay_rate,
            'recency_window_hours': self.recency_window_hours
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load user profile from file.
        
        Args:
            filepath: Path to load the profile from
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.user_id = data.get('user_id', self.user_id)
        self.max_history_size = data.get('max_history_size', 1000)
        self.history_decay_rate = data.get('history_decay_rate', 0.05)
        self.recency_window_hours = data.get('recency_window_hours', 24)
        
        # Load search history
        self.search_history = []
        for entry_data in data.get('search_history', []):
            self.search_history.append(SearchHistoryEntry.from_dict(entry_data))
        
        # Load preferences
        self.interest_categories = defaultdict(float, data.get('interest_categories', {}))
        self.keyword_preferences = defaultdict(float, data.get('keyword_preferences', {}))
        self.topic_preferences = defaultdict(float, data.get('topic_preferences', {}))
        
        # Load feedback data
        self.click_through_data = defaultdict(lambda: defaultdict(int))
        for query, doc_clicks in data.get('click_through_data', {}).items():
            self.click_through_data[query] = defaultdict(int, doc_clicks)
        
        self.relevance_feedback = data.get('relevance_feedback', {})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get user profile statistics."""
        recent_searches = [
            entry for entry in self.search_history
            if (datetime.now() - entry.timestamp).days <= 7
        ]
        
        return {
            'user_id': self.user_id,
            'total_searches': len(self.search_history),
            'recent_searches': len(recent_searches),
            'top_categories': sorted(
                self.interest_categories.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'top_keywords': sorted(
                self.keyword_preferences.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'feedback_entries': len(self.relevance_feedback)
        }