"""
Utility functions for the information retrieval system.
"""

import re
import string
from typing import List, Dict, Any
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    stopwords = None
    word_tokenize = None
    NLTK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Download required NLTK data
if NLTK_AVAILABLE and nltk:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

class TextPreprocessor:
    """Text preprocessing utilities for IR system."""
    
    def __init__(self):
        if NLTK_AVAILABLE and stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            # Fallback stopwords
            self.stop_words = set([
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
            ])
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning, tokenizing, and removing stopwords.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text string
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        if NLTK_AVAILABLE and word_tokenize:
            tokens = word_tokenize(text)
        else:
            # Simple tokenization fallback
            tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        processed_text = self.preprocess_text(text)
        tokens = processed_text.split()
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
            
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]


class SimilarityCalculator:
    """Utilities for calculating various similarity metrics."""
    
    @staticmethod
    def cosine_similarity(vec1, vec2) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        if NUMPY_AVAILABLE and np:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
        else:
            # Fallback calculation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = (sum(x * x for x in vec1)) ** 0.5
            norm2 = (sum(x * x for x in vec2)) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def euclidean_distance(vec1, vec2) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Euclidean distance
        """
        if NUMPY_AVAILABLE and np:
            return float(np.linalg.norm(vec1 - vec2))
        else:
            # Fallback calculation
            return (sum((a - b) ** 2 for a, b in zip(vec1, vec2))) ** 0.5
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity score
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class TemporalWeighting:
    """Utilities for temporal weighting of search history."""
    
    @staticmethod
    def exponential_decay(time_diff_hours: float, decay_rate: float = 0.1) -> float:
        """
        Calculate exponential decay weight based on time difference.
        
        Args:
            time_diff_hours: Time difference in hours
            decay_rate: Rate of decay (higher = faster decay)
            
        Returns:
            Weight between 0 and 1
        """
        if NUMPY_AVAILABLE and np:
            return float(np.exp(-decay_rate * time_diff_hours))
        else:
            import math
            return math.exp(-decay_rate * time_diff_hours)
    
    @staticmethod
    def linear_decay(time_diff_hours: float, max_hours: float = 168) -> float:
        """
        Calculate linear decay weight based on time difference.
        
        Args:
            time_diff_hours: Time difference in hours
            max_hours: Maximum hours for consideration (default: 1 week)
            
        Returns:
            Weight between 0 and 1
        """
        if time_diff_hours >= max_hours:
            return 0.0
        return 1.0 - (time_diff_hours / max_hours)
    
    @staticmethod
    def recency_boost(time_diff_hours: float, boost_window: float = 24) -> float:
        """
        Apply recency boost to recent searches.
        
        Args:
            time_diff_hours: Time difference in hours
            boost_window: Window for applying boost (default: 24 hours)
            
        Returns:
            Boost multiplier
        """
        if time_diff_hours <= boost_window:
            return 1.5 - (time_diff_hours / boost_window) * 0.5
        return 1.0


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to [0, 1] range.
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
        
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
        
    return [(score - min_score) / (max_score - min_score) for score in scores]


def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average of values.
    
    Args:
        values: List of values
        weights: List of weights
        
    Returns:
        Weighted average
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0
        
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
        
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def get_topic_categories() -> Dict[str, List[str]]:
    """
    Get predefined topic categories with associated keywords.
    
    Returns:
        Dictionary mapping category names to keyword lists
    """
    return {
        'technology': [
            'programming', 'software', 'computer', 'algorithm', 'code', 
            'development', 'framework', 'library', 'api', 'database',
            'machine learning', 'artificial intelligence', 'data science',
            'web development', 'mobile app', 'cloud computing'
        ],
        'science': [
            'research', 'experiment', 'hypothesis', 'theory', 'physics',
            'chemistry', 'biology', 'mathematics', 'statistics', 'analysis',
            'laboratory', 'scientific', 'method', 'discovery', 'innovation'
        ],
        'biology': [
            'animal', 'plant', 'species', 'ecosystem', 'habitat', 'evolution',
            'genetics', 'dna', 'organism', 'biodiversity', 'conservation',
            'wildlife', 'nature', 'environment', 'cell', 'molecular'
        ],
        'history': [
            'ancient', 'medieval', 'war', 'empire', 'civilization', 'culture',
            'historical', 'century', 'period', 'timeline', 'archaeology',
            'heritage', 'tradition', 'revolution', 'dynasty', 'monument'
        ],
        'sports': [
            'game', 'team', 'player', 'match', 'championship', 'tournament',
            'athletic', 'competition', 'sport', 'training', 'fitness',
            'exercise', 'olympic', 'professional', 'league', 'stadium'
        ],
        'entertainment': [
            'movie', 'film', 'music', 'song', 'artist', 'actor', 'director',
            'entertainment', 'show', 'television', 'streaming', 'performance',
            'concert', 'album', 'celebrity', 'award', 'festival'
        ],
        'health': [
            'medical', 'medicine', 'health', 'disease', 'treatment', 'therapy',
            'doctor', 'hospital', 'patient', 'diagnosis', 'symptom',
            'healthcare', 'wellness', 'nutrition', 'exercise', 'mental health'
        ],
        'business': [
            'company', 'business', 'market', 'economy', 'finance', 'investment',
            'management', 'strategy', 'corporate', 'entrepreneur', 'startup',
            'revenue', 'profit', 'marketing', 'sales', 'industry'
        ]
    }