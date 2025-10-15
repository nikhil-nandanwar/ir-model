"""
Context-aware information retrieval engine that personalizes search results
based on user search history and preferences.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict

from .document_index import DocumentIndex, Document
from .user_profile import UserProfile
from .utils import normalize_scores, weighted_average, SimilarityCalculator


class SearchResult:
    """Represents a search result with scoring information."""
    
    def __init__(self, document: Document, score: float, 
                 score_components: Dict[str, float] = None):
        self.document = document
        self.score = score
        self.score_components = score_components or {}
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'doc_id': self.document.doc_id,
            'title': self.document.title,
            'content': self.document.content[:200] + "..." if len(self.document.content) > 200 else self.document.content,
            'categories': self.document.categories,
            'score': self.score,
            'score_components': self.score_components
        }


class ContextAwareRetriever:
    """Main retrieval engine with context awareness."""
    
    def __init__(self, document_index: DocumentIndex, user_profile: UserProfile = None):
        self.document_index = document_index
        self.user_profile = user_profile
        self.similarity_calc = SimilarityCalculator()
        
        # Scoring weights (can be tuned)
        self.weights = {
            'tfidf_similarity': 0.3,
            'bert_similarity': 0.3,
            'user_preference': 0.2,
            'category_match': 0.1,
            'feedback_score': 0.1
        }
        
        # Query processing settings
        self.max_expansion_terms = 3
        self.min_similarity_threshold = 0.01
        
    def search(self, query: str, max_results: int = 10, 
               use_context: bool = True) -> List[SearchResult]:
        """
        Perform context-aware search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            use_context: Whether to use user context for personalization
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        if not query.strip():
            return []
        
        # Step 1: Query expansion based on user history
        expanded_query = self._expand_query(query) if use_context and self.user_profile else query
        
        # Step 2: Get candidate documents
        candidate_docs = self._get_candidate_documents(expanded_query)
        
        if not candidate_docs:
            return []
        
        # Step 3: Calculate various similarity scores
        scores = self._calculate_scores(query, expanded_query, candidate_docs, use_context)
        
        # Step 4: Combine scores and rank
        final_scores = self._combine_scores(scores, candidate_docs)
        
        # Step 5: Sort and return top results
        ranked_results = self._create_ranked_results(final_scores, max_results)
        
        # Step 6: Log search to user profile
        if self.user_profile and use_context:
            clicked_docs = []  # Will be updated later with user interactions
            categories = self._infer_query_categories(query)
            self.user_profile.add_search(query, clicked_docs, categories)
        
        return ranked_results
    
    def search_with_filters(self, query: str, categories: List[str] = None,
                          date_range: Tuple[datetime, datetime] = None,
                          max_results: int = 10) -> List[SearchResult]:
        """
        Search with additional filters.
        
        Args:
            query: Search query
            categories: Categories to filter by
            date_range: Date range to filter by (start, end)
            max_results: Maximum results to return
            
        Returns:
            Filtered search results
        """
        # Get all results first
        results = self.search(query, max_results * 2)  # Get more to allow filtering
        
        filtered_results = []
        
        for result in results:
            doc = result.document
            
            # Apply category filter
            if categories and not any(cat in doc.categories for cat in categories):
                continue
            
            # Apply date filter
            if date_range:
                start_date, end_date = date_range
                if not (start_date <= doc.created_at <= end_date):
                    continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= max_results:
                break
        
        return filtered_results
    
    def get_recommendations(self, max_results: int = 10) -> List[SearchResult]:
        """
        Get document recommendations based on user profile.
        
        Args:
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended documents
        """
        if not self.user_profile:
            return []
        
        # Get user interest profile
        interest_profile = self.user_profile.get_interest_profile()
        
        # Score all documents based on user preferences
        recommendations = []
        
        for doc_id, doc in self.document_index.documents.items():
            preference_score = self.user_profile.calculate_document_preference_score(
                doc.categories, doc.keywords
            )
            
            if preference_score > 0.1:  # Minimum threshold
                result = SearchResult(doc, preference_score, {'user_preference': preference_score})
                recommendations.append(result)
        
        # Sort by preference score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations[:max_results]
    
    def update_user_feedback(self, query: str, doc_id: str, clicked: bool) -> None:
        """
        Update user feedback for learning.
        
        Args:
            query: Search query
            doc_id: Document ID
            clicked: Whether user clicked on the document
        """
        if self.user_profile:
            self.user_profile.update_click_feedback(query, doc_id, clicked)
    
    def add_document(self, doc_id: str, title: str, content: str,
                    categories: List[str] = None, metadata: Dict = None) -> None:
        """
        Add a document to the search index.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            content: Document content
            categories: List of categories
            metadata: Additional metadata
        """
        self.document_index.add_document(doc_id, title, content, categories, metadata)
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query based on user search history.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query string
        """
        if not self.user_profile:
            return query
        
        expansion_terms = self.user_profile.get_query_expansion_terms(
            query, self.max_expansion_terms
        )
        
        if expansion_terms:
            return f"{query} {' '.join(expansion_terms)}"
        
        return query
    
    def _get_candidate_documents(self, query: str) -> List[str]:
        """
        Get candidate documents using multiple retrieval methods.
        
        Args:
            query: Search query (possibly expanded)
            
        Returns:
            List of candidate document IDs
        """
        candidates = set()
        
        # Method 1: Keyword-based search
        from .utils import TextPreprocessor
        preprocessor = TextPreprocessor()
        keywords = preprocessor.extract_keywords(query)
        keyword_candidates = self.document_index.search_keywords(keywords)
        candidates.update(keyword_candidates)
        
        # Method 2: Category-based search (if we can infer categories)
        categories = self._infer_query_categories(query)
        if categories:
            category_candidates = self.document_index.search_categories(categories)
            candidates.update(category_candidates)
        
        # Method 3: If no candidates found, use all documents (for small collections)
        if not candidates and self.document_index.get_document_count() < 1000:
            candidates = set(self.document_index.documents.keys())
        
        return list(candidates)
    
    def _calculate_scores(self, original_query: str, expanded_query: str, 
                         candidate_docs: List[str], use_context: bool) -> Dict[str, Dict[str, float]]:
        """
        Calculate various similarity scores for candidate documents.
        
        Args:
            original_query: Original user query
            expanded_query: Query with expansion terms
            candidate_docs: List of candidate document IDs
            use_context: Whether to use user context
            
        Returns:
            Dictionary of score types to document scores
        """
        scores = {}
        
        # TF-IDF similarity
        tfidf_scores = self.document_index.calculate_tfidf_similarity(
            expanded_query, candidate_docs
        )
        scores['tfidf_similarity'] = tfidf_scores
        
        # BERT similarity
        bert_scores = self.document_index.calculate_bert_similarity(
            expanded_query, candidate_docs
        )
        scores['bert_similarity'] = bert_scores
        
        # User preference scores (if context is available)
        if use_context and self.user_profile:
            preference_scores = {}
            for doc_id in candidate_docs:
                doc = self.document_index.get_document(doc_id)
                if doc:
                    score = self.user_profile.calculate_document_preference_score(
                        doc.categories, doc.keywords
                    )
                    preference_scores[doc_id] = score
            scores['user_preference'] = preference_scores
        else:
            scores['user_preference'] = {doc_id: 0.0 for doc_id in candidate_docs}
        
        # Category match scores
        category_scores = self._calculate_category_scores(original_query, candidate_docs)
        scores['category_match'] = category_scores
        
        # Feedback scores (if available)
        if use_context and self.user_profile:
            feedback_scores = {}
            for doc_id in candidate_docs:
                score = self.user_profile.get_relevance_score(original_query, doc_id)
                feedback_scores[doc_id] = max(0, score + 0.5)  # Normalize to positive
            scores['feedback_score'] = feedback_scores
        else:
            scores['feedback_score'] = {doc_id: 0.5 for doc_id in candidate_docs}
        
        return scores
    
    def _calculate_category_scores(self, query: str, candidate_docs: List[str]) -> Dict[str, float]:
        """
        Calculate category-based similarity scores.
        
        Args:
            query: Search query
            candidate_docs: Candidate document IDs
            
        Returns:
            Category match scores
        """
        query_categories = self._infer_query_categories(query)
        category_scores = {}
        
        for doc_id in candidate_docs:
            doc = self.document_index.get_document(doc_id)
            if doc and query_categories:
                # Calculate category overlap
                doc_categories = set(doc.categories)
                query_categories_set = set(query_categories)
                
                if doc_categories and query_categories_set:
                    overlap = len(doc_categories.intersection(query_categories_set))
                    total = len(doc_categories.union(query_categories_set))
                    score = overlap / total if total > 0 else 0.0
                else:
                    score = 0.0
            else:
                score = 0.0
            
            category_scores[doc_id] = score
        
        return category_scores
    
    def _infer_query_categories(self, query: str) -> List[str]:
        """
        Infer categories from query text.
        
        Args:
            query: Search query
            
        Returns:
            List of inferred categories
        """
        from .utils import get_topic_categories
        
        query_lower = query.lower()
        topic_categories = get_topic_categories()
        inferred_categories = []
        
        for category, keywords in topic_categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    inferred_categories.append(category)
                    break
        
        return list(set(inferred_categories))  # Remove duplicates
    
    def _combine_scores(self, scores: Dict[str, Dict[str, float]], 
                       candidate_docs: List[str]) -> Dict[str, Tuple[float, Dict[str, float]]]:
        """
        Combine different score components into final scores.
        
        Args:
            scores: Dictionary of score components
            candidate_docs: List of candidate document IDs
            
        Returns:
            Dictionary mapping doc_id to (final_score, score_components)
        """
        final_scores = {}
        
        # Compute topical context boost from recent user history if available
        context_topic_boosts = {}
        if self.user_profile:
            # Derive top topics from recent history
            recent_topics = sorted(self.user_profile.topic_preferences.items(), key=lambda x: x[1], reverse=True)
            top_topics = [t for t, v in recent_topics[:3] if v > 0]
        else:
            top_topics = []

        for doc_id in candidate_docs:
            # Collect individual scores
            score_components = {}
            for score_type, doc_scores in scores.items():
                score_components[score_type] = doc_scores.get(doc_id, 0.0)
            
            # Calculate weighted combination
            weighted_score = 0.0
            for score_type, weight in self.weights.items():
                if score_type in score_components:
                    weighted_score += score_components[score_type] * weight

            # Apply context-topic boost: if document matches user's recent top topics
            context_boost = 0.0
            if top_topics:
                doc = self.document_index.get_document(doc_id)
                if doc:
                    for t in top_topics:
                        if t in doc.categories or t in doc.keywords:
                            # boost proportional to user's preference strength for the topic
                            pref_strength = self.user_profile.topic_preferences.get(t, 0.0)
                            context_boost += 0.1 * pref_strength

            # Cap context boost and add to weighted score
            context_boost = min(context_boost, 0.5)
            final_score_with_context = weighted_score + context_boost

            score_components['context_topic_boost'] = context_boost
            final_scores[doc_id] = (final_score_with_context, score_components)
        
        return final_scores
    
    def _create_ranked_results(self, final_scores: Dict[str, Tuple[float, Dict[str, float]]], 
                             max_results: int) -> List[SearchResult]:
        """
        Create ranked search results.
        
        Args:
            final_scores: Final scores with components
            max_results: Maximum number of results
            
        Returns:
            List of SearchResult objects
        """
        # Sort by final score
        sorted_docs = sorted(
            final_scores.items(), 
            key=lambda x: x[1][0], 
            reverse=True
        )
        
        results = []
        for doc_id, (score, components) in sorted_docs[:max_results]:
            doc = self.document_index.get_document(doc_id)
            if doc and score >= self.min_similarity_threshold:
                result = SearchResult(doc, score, components)
                results.append(result)
        
        return results
    
    def set_scoring_weights(self, weights: Dict[str, float]) -> None:
        """
        Update scoring weights.
        
        Args:
            weights: Dictionary of weight updates
        """
        for weight_type, weight_value in weights.items():
            if weight_type in self.weights:
                self.weights[weight_type] = weight_value
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search system statistics."""
        stats = {
            'document_index': self.document_index.get_statistics(),
            'scoring_weights': self.weights.copy(),
            'has_user_profile': self.user_profile is not None
        }
        
        if self.user_profile:
            stats['user_profile'] = self.user_profile.get_statistics()
        
        return stats