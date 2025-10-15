#!/usr/bin/env python3
"""
Comprehensive demonstration of the Context-Aware Information Retrieval System.

This script demonstrates the key features of the system including:
1. Document indexing and retrieval
2. User profile creation and management
3. Context-aware search personalization
4. Comparison between baseline and personalized results
5. Visualization of ranking changes

Usage:
    python examples/demo.py
"""

import os
import sys
import json
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.document_index import DocumentIndex
    from src.user_profile import UserProfile
    from src.retriever import ContextAwareRetriever
    from visualization.ranking_analysis import IRVisualizationTools
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run this script from the project root directory")
    sys.exit(1)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_subheader(title):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def load_sample_documents():
    """Load sample documents from JSON file."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_documents.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Sample documents file not found. Creating minimal test data...")
        return [
            {
                'doc_id': 'python_prog',
                'title': 'Python Programming Language',
                'content': 'Python is a high-level programming language used for software development, data science, and automation.',
                'categories': ['technology', 'programming']
            },
            {
                'doc_id': 'python_snake',
                'title': 'Python Snake',
                'content': 'Python snakes are large, non-venomous snakes found in Africa, Asia, and Australia.',
                'categories': ['biology', 'animals']
            },
            {
                'doc_id': 'machine_learning',
                'title': 'Machine Learning Basics',
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.',
                'categories': ['technology', 'artificial_intelligence']
            }
        ]


def setup_document_index():
    """Initialize and populate the document index."""
    print_header("SETTING UP DOCUMENT INDEX")
    
    # Initialize document index
    doc_index = DocumentIndex(use_bert=False)  # Disable BERT for faster demo
    print("✓ Document index initialized")
    
    # Load sample documents
    documents = load_sample_documents()
    print(f"✓ Loaded {len(documents)} sample documents")
    
    # Add documents to index
    for doc in documents:
        doc_index.add_document(
            doc['doc_id'],
            doc['title'],
            doc['content'],
            doc.get('categories', []),
            doc.get('metadata', {})
        )
    
    print("✓ Documents added to index")
    
    # Display index statistics
    stats = doc_index.get_statistics()
    print(f"\nIndex Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total categories: {stats['total_categories']}")
    print(f"  Categories: {', '.join(stats['categories'])}")
    
    return doc_index


def create_user_profiles():
    """Create different types of user profiles for demonstration."""
    print_header("CREATING USER PROFILES")
    
    # Tech-oriented user
    tech_user = UserProfile("tech_user")
    tech_searches = [
        ("python programming tutorial", ["technology", "programming"]),
        ("machine learning algorithms", ["technology", "artificial_intelligence"]),
        ("software development best practices", ["technology", "programming"]),
        ("database optimization techniques", ["technology", "databases"]),
        ("web development frameworks", ["technology", "web"]),
    ]
    
    print_subheader("Tech User Profile")
    for query, categories in tech_searches:
        tech_user.add_search(query, [], categories)
        print(f"  Added search: '{query}'")
    
    # Biology-oriented user
    bio_user = UserProfile("bio_user")
    bio_searches = [
        ("animal behavior research", ["biology", "animals"]),
        ("ecosystem conservation methods", ["biology", "conservation"]),
        ("reptile species classification", ["biology", "animals"]),
        ("marine biology studies", ["biology", "marine"]),
        ("genetic diversity analysis", ["biology", "genetics"]),
    ]
    
    print_subheader("Biology User Profile")
    for query, categories in bio_searches:
        bio_user.add_search(query, [], categories)
        print(f"  Added search: '{query}'")
    
    # History-oriented user
    history_user = UserProfile("history_user")
    history_searches = [
        ("ancient civilizations", ["history", "ancient"]),
        ("world war timeline", ["history", "war"]),
        ("renaissance art movements", ["history", "art"]),
        ("archaeological discoveries", ["history", "archaeology"]),
        ("cultural heritage preservation", ["history", "culture"]),
    ]
    
    print_subheader("History User Profile")
    for query, categories in history_searches:
        history_user.add_search(query, [], categories)
        print(f"  Added search: '{query}'")
    
    return {
        'tech_user': tech_user,
        'bio_user': bio_user,
        'history_user': history_user
    }


def demonstrate_search_comparison(doc_index, user_profiles, query="python"):
    """Demonstrate search result differences between users."""
    print_header(f"SEARCH COMPARISON FOR QUERY: '{query}'")
    
    # Baseline search (no personalization)
    baseline_retriever = ContextAwareRetriever(doc_index, None)
    baseline_results = baseline_retriever.search(query, max_results=5, use_context=False)
    
    print_subheader("Baseline Results (No Personalization)")
    if baseline_results:
        for i, result in enumerate(baseline_results, 1):
            print(f"  {i}. {result.document.title}")
            print(f"     Score: {result.score:.4f}")
            print(f"     Categories: {', '.join(result.document.categories)}")
    else:
        print("  No results found")
    
    # Compare with different user profiles
    for user_type, user_profile in user_profiles.items():
        print_subheader(f"{user_type.title().replace('_', ' ')} - Personalized Results")
        
        personalized_retriever = ContextAwareRetriever(doc_index, user_profile)
        personalized_results = personalized_retriever.search(query, max_results=5, use_context=True)
        
        if personalized_results:
            for i, result in enumerate(personalized_results, 1):
                # Check if ranking changed from baseline
                baseline_rank = None
                for j, baseline_result in enumerate(baseline_results):
                    if baseline_result.document.doc_id == result.document.doc_id:
                        baseline_rank = j + 1
                        break
                
                rank_change = ""
                if baseline_rank:
                    if i < baseline_rank:
                        rank_change = f" (↑ from #{baseline_rank})"
                    elif i > baseline_rank:
                        rank_change = f" (↓ from #{baseline_rank})"
                    else:
                        rank_change = f" (same as #{baseline_rank})"
                else:
                    rank_change = " (new in results)"
                
                print(f"  {i}. {result.document.title}{rank_change}")
                print(f"     Score: {result.score:.4f}")
                print(f"     Categories: {', '.join(result.document.categories)}")
                
                # Show score components if available
                if result.score_components:
                    components = []
                    for comp, score in result.score_components.items():
                        if score > 0.01:  # Only show significant components
                            components.append(f"{comp}: {score:.3f}")
                    if components:
                        print(f"     Components: {', '.join(components)}")
        else:
            print("  No results found")


def demonstrate_user_profile_analysis(user_profiles):
    """Analyze and display user profile characteristics."""
    print_header("USER PROFILE ANALYSIS")
    
    for user_type, user_profile in user_profiles.items():
        print_subheader(f"{user_type.title().replace('_', ' ')} Profile Analysis")
        
        stats = user_profile.get_statistics()
        interest_profile = user_profile.get_interest_profile()
        
        print(f"  Total searches: {stats['total_searches']}")
        print(f"  Recent searches: {stats['recent_searches']}")
        
        # Top categories
        if stats['top_categories']:
            print("  Top interest categories:")
            for category, score in stats['top_categories'][:3]:
                print(f"    - {category}: {score:.2f}")
        
        # Top keywords  
        if stats['top_keywords']:
            print("  Top keywords:")
            for keyword, score in stats['top_keywords'][:5]:
                print(f"    - {keyword}: {score:.2f}")
        
        # Query expansion example
        expansion_terms = user_profile.get_query_expansion_terms("search", max_terms=3)
        if expansion_terms:
            print(f"  Query expansion for 'search': {', '.join(expansion_terms)}")


def demonstrate_feedback_learning(doc_index, user_profiles):
    """Demonstrate how user feedback affects future searches."""
    print_header("FEEDBACK LEARNING DEMONSTRATION")
    
    tech_user = user_profiles['tech_user']
    retriever = ContextAwareRetriever(doc_index, tech_user)
    
    # Initial search
    print_subheader("Initial Search for 'programming'")
    initial_results = retriever.search("programming", max_results=3)
    
    if initial_results:
        for i, result in enumerate(initial_results, 1):
            print(f"  {i}. {result.document.title} (Score: {result.score:.4f})")
    
    # Simulate clicking on a specific document
    print_subheader("Simulating User Feedback")
    if initial_results:
        clicked_doc = initial_results[0].document.doc_id
        print(f"  User clicks on: {initial_results[0].document.title}")
        
        # Update feedback
        retriever.update_user_feedback("programming", clicked_doc, True)
        print("  ✓ Positive feedback recorded")
        
        # Search again to show effect
        print_subheader("Search After Feedback")
        updated_results = retriever.search("programming", max_results=3)
        
        if updated_results:
            for i, result in enumerate(updated_results, 1):
                feedback_score = tech_user.get_relevance_score("programming", result.document.doc_id)
                feedback_indicator = ""
                if feedback_score > 0:
                    feedback_indicator = " 👍"
                elif feedback_score < 0:
                    feedback_indicator = " 👎"
                
                print(f"  {i}. {result.document.title} (Score: {result.score:.4f}){feedback_indicator}")


def demonstrate_temporal_effects(user_profiles):
    """Demonstrate how temporal weighting affects search history influence."""
    print_header("TEMPORAL WEIGHTING DEMONSTRATION")
    
    tech_user = user_profiles['tech_user']
    
    # Add some older searches with backdated timestamps
    older_time = datetime.now() - timedelta(days=7)
    recent_time = datetime.now() - timedelta(hours=2)
    
    print_subheader("Adding Time-Weighted Search History")
    print(f"  Adding older search (7 days ago): 'database systems'")
    print(f"  Adding recent search (2 hours ago): 'neural networks'")
    
    # Simulate different timestamps (would normally be handled by the system)
    from src.user_profile import SearchHistoryEntry
    
    # Add an older search
    older_entry = SearchHistoryEntry("database systems", older_time, [], ['technology', 'databases'])
    older_entry.processed_query = "database systems"
    tech_user.search_history.append(older_entry)
    
    # Add a recent search
    recent_entry = SearchHistoryEntry("neural networks", recent_time, [], ['technology', 'artificial_intelligence'])
    recent_entry.processed_query = "neural networks"
    tech_user.search_history.append(recent_entry)
    
    # Get context vector to show temporal weighting
    context_vector = tech_user.get_user_context_vector()
    
    print_subheader("Context Vector Analysis")
    print("  Weighted terms from search history:")
    
    # Sort context items by weight
    sorted_context = sorted(context_vector.items(), key=lambda x: x[1], reverse=True)
    for term, weight in sorted_context[:8]:
        if weight > 0.1:  # Only show significant weights
            print(f"    {term}: {weight:.3f}")


def demonstrate_query_expansion(user_profiles):
    """Demonstrate query expansion based on user history."""
    print_header("QUERY EXPANSION DEMONSTRATION")
    
    for user_type, user_profile in user_profiles.items():
        print_subheader(f"{user_type.title().replace('_', ' ')} Query Expansion")
        
        test_queries = ["research", "analysis", "development"]
        
        for query in test_queries:
            expansion_terms = user_profile.get_query_expansion_terms(query, max_terms=3)
            if expansion_terms:
                expanded_query = f"{query} {' '.join(expansion_terms)}"
                print(f"  '{query}' → '{expanded_query}'")
            else:
                print(f"  '{query}' → no expansion")


def save_demonstration_results(doc_index, user_profiles):
    """Save demonstration data for further analysis."""
    print_header("SAVING DEMONSTRATION RESULTS")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'demo_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save document index
    index_path = os.path.join(output_dir, 'document_index.json')
    doc_index.save_to_file(index_path)
    print(f"✓ Document index saved to: {index_path}")
    
    # Save user profiles
    for user_type, user_profile in user_profiles.items():
        profile_path = os.path.join(output_dir, f'{user_type}_profile.json')
        user_profile.save_to_file(profile_path)
        print(f"✓ {user_type} profile saved to: {profile_path}")
    
    # Generate summary report
    report_path = os.path.join(output_dir, 'demo_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CONTEXT-AWARE INFORMATION RETRIEVAL SYSTEM\n")
        f.write("Demo Summary Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        f.write("SYSTEM STATISTICS:\n")
        stats = doc_index.get_statistics()
        f.write(f"Total Documents: {stats['total_documents']}\n")
        f.write(f"Categories: {', '.join(stats['categories'])}\n")
        f.write(f"Uses BERT: {stats['use_bert']}\n")
        f.write(f"Has TF-IDF: {stats['has_tfidf']}\n\n")
        
        f.write("USER PROFILES:\n")
        for user_type, user_profile in user_profiles.items():
            user_stats = user_profile.get_statistics()
            f.write(f"\n{user_type.upper()}:\n")
            f.write(f"  Total Searches: {user_stats['total_searches']}\n")
            f.write(f"  Top Categories: {[cat for cat, _ in user_stats['top_categories'][:3]]}\n")
            f.write(f"  Top Keywords: {[kw for kw, _ in user_stats['top_keywords'][:5]]}\n")
    
    print(f"✓ Demo summary saved to: {report_path}")
    print(f"\nAll demonstration files saved in: {output_dir}")


def main():
    """Run the complete demonstration."""
    print("CONTEXT-AWARE INFORMATION RETRIEVAL SYSTEM")
    print("Complete System Demonstration")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Setup document index
        doc_index = setup_document_index()
        
        # Step 2: Create user profiles
        user_profiles = create_user_profiles()
        
        # Step 3: Demonstrate search comparisons
        demonstrate_search_comparison(doc_index, user_profiles, "python")
        demonstrate_search_comparison(doc_index, user_profiles, "research")
        
        # Step 4: Analyze user profiles
        demonstrate_user_profile_analysis(user_profiles)
        
        # Step 5: Show feedback learning
        demonstrate_feedback_learning(doc_index, user_profiles)
        
        # Step 6: Demonstrate temporal effects
        demonstrate_temporal_effects(user_profiles)
        
        # Step 7: Show query expansion
        demonstrate_query_expansion(user_profiles)
        
        # Step 8: Save results
        save_demonstration_results(doc_index, user_profiles)
        
        print_header("DEMONSTRATION COMPLETE")
        print("✓ All features successfully demonstrated")
        print("\nNext Steps:")
        print("1. Run 'python web/app.py' to start the web interface")
        print("2. Check the 'examples/demo_output/' directory for saved results")
        print("3. Try different queries in the web interface to see personalization")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())