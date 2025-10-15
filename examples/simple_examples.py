#!/usr/bin/env python3
"""
Simple usage examples for the Context-Aware Information Retrieval System.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_index import DocumentIndex
from src.user_profile import UserProfile  
from src.retriever import ContextAwareRetriever


def basic_usage_example():
    """Basic usage example showing core functionality."""
    print("=== BASIC USAGE EXAMPLE ===\n")
    
    # 1. Create document index
    print("1. Creating document index...")
    doc_index = DocumentIndex(use_bert=False)
    
    # 2. Add some documents
    print("2. Adding documents...")
    documents = [
        {
            'id': 'doc1',
            'title': 'Python Programming Guide',
            'content': 'Python is a versatile programming language used for web development, data science, and automation.',
            'categories': ['technology', 'programming']
        },
        {
            'id': 'doc2', 
            'title': 'Python Snake Species',
            'content': 'Pythons are large, non-venomous snakes found in tropical regions of Africa, Asia, and Australia.',
            'categories': ['biology', 'animals']
        },
        {
            'id': 'doc3',
            'title': 'Machine Learning with Python',
            'content': 'Machine learning libraries in Python like scikit-learn and TensorFlow make AI development accessible.',
            'categories': ['technology', 'artificial_intelligence']
        }
    ]
    
    for doc in documents:
        doc_index.add_document(doc['id'], doc['title'], doc['content'], doc['categories'])
    
    print(f"   Added {len(documents)} documents")
    
    # 3. Create user profile
    print("3. Creating user profile...")
    user_profile = UserProfile("example_user")
    
    # 4. Simulate some search history
    print("4. Building search history...")
    search_history = [
        ("machine learning tutorial", ["technology", "programming"]),
        ("python data analysis", ["technology", "programming"]),
        ("software development", ["technology", "programming"])
    ]
    
    for query, categories in search_history:
        user_profile.add_search(query, [], categories)
    
    # 5. Create retriever
    print("5. Setting up retriever...")
    retriever = ContextAwareRetriever(doc_index, user_profile)
    
    # 6. Perform searches
    print("6. Performing searches...\n")
    
    # Search without personalization
    print("Search for 'python' (baseline):")
    baseline_retriever = ContextAwareRetriever(doc_index, None)
    baseline_results = baseline_retriever.search("python", use_context=False)
    
    for i, result in enumerate(baseline_results, 1):
        print(f"   {i}. {result.document.title} (Score: {result.score:.3f})")
    
    print("\nSearch for 'python' (personalized):")
    personalized_results = retriever.search("python", use_context=True)
    
    for i, result in enumerate(personalized_results, 1):
        print(f"   {i}. {result.document.title} (Score: {result.score:.3f})")
    
    print("\n✓ Basic example completed!")


def advanced_usage_example():
    """Advanced usage showing more features."""
    print("\n=== ADVANCED USAGE EXAMPLE ===\n")
    
    # Setup (reuse from basic example)
    doc_index = DocumentIndex(use_bert=False)
    
    # Add more diverse documents
    advanced_docs = [
        {
            'id': 'tech1',
            'title': 'Neural Networks Fundamentals',
            'content': 'Neural networks are computing systems inspired by biological neural networks in animal brains.',
            'categories': ['technology', 'artificial_intelligence']
        },
        {
            'id': 'bio1',
            'title': 'Ecosystem Conservation',
            'content': 'Ecosystem conservation involves protecting natural habitats and biodiversity for future generations.',
            'categories': ['biology', 'environment', 'conservation']
        },
        {
            'id': 'hist1',
            'title': 'Ancient Roman Engineering',
            'content': 'Roman engineers built impressive structures like aqueducts, roads, and amphitheaters that lasted centuries.',
            'categories': ['history', 'engineering', 'ancient']
        }
    ]
    
    for doc in advanced_docs:
        doc_index.add_document(doc['id'], doc['title'], doc['content'], doc['categories'])
    
    # Create different user types
    print("1. Creating specialized user profiles...")
    
    # Tech enthusiast
    tech_user = UserProfile("tech_enthusiast")
    tech_searches = [
        "machine learning algorithms",
        "neural network architecture", 
        "artificial intelligence applications",
        "deep learning frameworks"
    ]
    
    for query in tech_searches:
        tech_user.add_search(query, [], ['technology', 'artificial_intelligence'])
    
    # Biology researcher  
    bio_user = UserProfile("biology_researcher")
    bio_searches = [
        "ecosystem biodiversity",
        "conservation strategies",
        "environmental protection",
        "species preservation"
    ]
    
    for query in bio_searches:
        bio_user.add_search(query, [], ['biology', 'environment'])
    
    # 2. Compare search results across users
    print("2. Comparing personalized results...\n")
    
    test_query = "network"
    
    print(f"Query: '{test_query}'\n")
    
    # Tech user results
    print("Tech User Results:")
    tech_retriever = ContextAwareRetriever(doc_index, tech_user)
    tech_results = tech_retriever.search(test_query)
    
    for i, result in enumerate(tech_results, 1):
        print(f"   {i}. {result.document.title}")
        print(f"      Categories: {', '.join(result.document.categories)}")
        print(f"      Score: {result.score:.3f}")
    
    print("\nBiology User Results:")
    bio_retriever = ContextAwareRetriever(doc_index, bio_user)
    bio_results = bio_retriever.search(test_query)
    
    for i, result in enumerate(bio_results, 1):
        print(f"   {i}. {result.document.title}")
        print(f"      Categories: {', '.join(result.document.categories)}")
        print(f"      Score: {result.score:.3f}")
    
    # 3. Demonstrate feedback learning
    print("\n3. Demonstrating feedback learning...")
    
    # Simulate user clicking on a result
    if tech_results:
        clicked_doc = tech_results[0].document.doc_id
        print(f"   User clicks on: {tech_results[0].document.title}")
        
        tech_retriever.update_user_feedback(test_query, clicked_doc, True)
        print("   ✓ Positive feedback recorded")
    
    # 4. Show query expansion
    print("\n4. Query expansion examples...")
    
    expansion_terms = tech_user.get_query_expansion_terms("learning", max_terms=3)
    if expansion_terms:
        print(f"   Tech user - 'learning' expanded to: learning {' '.join(expansion_terms)}")
    
    expansion_terms = bio_user.get_query_expansion_terms("research", max_terms=3)
    if expansion_terms:
        print(f"   Bio user - 'research' expanded to: research {' '.join(expansion_terms)}")
    
    # 5. Get recommendations
    print("\n5. Personalized recommendations...")
    
    tech_recommendations = tech_retriever.get_recommendations(max_results=3)
    print("   Tech user recommendations:")
    for i, rec in enumerate(tech_recommendations, 1):
        print(f"      {i}. {rec.document.title} (Relevance: {rec.score:.3f})")
    
    print("\n✓ Advanced example completed!")


def performance_test():
    """Simple performance test with larger dataset."""
    print("\n=== PERFORMANCE TEST ===\n")
    
    import time
    
    # Create larger dataset
    print("1. Creating larger document set...")
    doc_index = DocumentIndex(use_bert=False)
    
    # Generate test documents
    categories_list = [
        ['technology', 'programming'],
        ['biology', 'research'],
        ['history', 'culture'],
        ['science', 'physics'],
        ['medicine', 'health']
    ]
    
    start_time = time.time()
    
    for i in range(100):  # Create 100 test documents
        category = categories_list[i % len(categories_list)]
        doc_index.add_document(
            f"doc_{i}",
            f"Test Document {i}",
            f"This is test document {i} about {' and '.join(category)} topics. " * 5,
            category
        )
    
    indexing_time = time.time() - start_time
    print(f"   ✓ Indexed 100 documents in {indexing_time:.2f} seconds")
    
    # Create user profile
    user_profile = UserProfile("test_user")
    for i in range(20):  # 20 search history entries
        user_profile.add_search(f"test query {i}", [], ['technology'])
    
    # Test search performance
    print("2. Testing search performance...")
    retriever = ContextAwareRetriever(doc_index, user_profile)
    
    search_times = []
    
    for i in range(10):  # 10 test searches
        start_time = time.time()
        results = retriever.search(f"test document {i}", max_results=10)
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"   ✓ Average search time: {avg_search_time:.4f} seconds")
    print(f"   ✓ Average results per search: {len(results)}")
    
    print("\n✓ Performance test completed!")


def main():
    """Run all examples."""
    print("CONTEXT-AWARE INFORMATION RETRIEVAL SYSTEM")
    print("Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        basic_usage_example()
        advanced_usage_example()
        performance_test()
        
        print("\n" + "=" * 50)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("• Run 'python examples/demo.py' for a complete demonstration")
        print("• Run 'python web/app.py' to start the web interface")
        print("• Explore the source code in the 'src/' directory")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())