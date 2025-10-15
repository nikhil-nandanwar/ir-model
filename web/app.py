"""
Flask web application for the context-aware information retrieval system.
"""

import os
import json
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import uuid

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.document_index import DocumentIndex
    from src.user_profile import UserProfile
    from src.retriever import ContextAwareRetriever
    from visualization.ranking_analysis import IRVisualizationTools
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the project root directory")

# Configure Flask to find templates/static relative to this file
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Use environment variable for secret key in production
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here-change-in-production')

# Global variables for the IR system
doc_index = None
retriever = None
user_profiles = {}
viz_tools = None

def initialize_system():
    """Initialize the IR system with sample data."""
    global doc_index, retriever, viz_tools
    
    # Initialize components
    doc_index = DocumentIndex(use_bert=True)
    viz_tools = IRVisualizationTools()
    
    # Load sample documents
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_documents.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        for doc in documents:
            doc_index.add_document(
                doc['doc_id'],
                doc['title'],
                doc['content'],
                doc.get('categories', []),
                doc.get('metadata', {})
            )
        
        print(f"Loaded {len(documents)} documents into the index")
        
    except FileNotFoundError:
        print("Sample documents not found. Creating minimal test data...")
        # Create some test documents
        test_docs = [
            {
                'doc_id': 'test1',
                'title': 'Python Programming',
                'content': 'Python is a programming language used for software development.',
                'categories': ['technology', 'programming']
            },
            {
                'doc_id': 'test2', 
                'title': 'Python Snake',
                'content': 'Python is a type of snake found in tropical regions.',
                'categories': ['biology', 'animals']
            }
        ]
        
        for doc in test_docs:
            doc_index.add_document(
                doc['doc_id'],
                doc['title'], 
                doc['content'],
                doc['categories']
            )


def get_user_profile(user_id):
    """Get or create user profile."""
    # Load from disk if available, otherwise create new
    profile_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_profiles')
    os.makedirs(profile_dir, exist_ok=True)
    profile_path = os.path.join(profile_dir, f"{user_id}.json")

    if user_id not in user_profiles:
        user_profiles[user_id] = UserProfile(user_id)
        # Attempt to load persisted profile
        try:
            user_profiles[user_id].load_from_file(profile_path)
        except Exception:
            # If loading fails, continue with fresh profile
            pass

    return user_profiles[user_id]


@app.route('/')
def home():
    """Home page with search interface."""
    # Generate a user session ID if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())[:8]
    
    user_profile = get_user_profile(session['user_id'])
    recent_searches = user_profile.search_history[-5:] if user_profile.search_history else []
    
    return render_template('index.html', 
                         user_id=session['user_id'],
                         recent_searches=recent_searches)


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    if not doc_index:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    query = data.get('query', '').strip()
    use_personalization = data.get('personalization', True)
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        user_id = session.get('user_id')
        user_profile = get_user_profile(user_id) if use_personalization else None
        
        # Create retriever with or without user profile
        retriever = ContextAwareRetriever(doc_index, user_profile)
        
        # Perform search
        results = retriever.search(query, max_results=10, use_context=use_personalization)
        
        # Convert results to JSON-serializable format
        search_results = []
        for result in results:
            search_results.append(result.to_dict())
        
        # Update user profile with search (if personalization is enabled)
        if use_personalization and user_profile:
            # retriever.search already logs the search into the in-memory profile.
            # Persist the updated profile to disk so history survives restarts.
            try:
                profile_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_profiles')
                os.makedirs(profile_dir, exist_ok=True)
                profile_path = os.path.join(profile_dir, f"{user_profile.user_id}.json")
                user_profile.save_to_file(profile_path)
            except Exception as e:
                print(f"Failed to save user profile: {e}")
        
        return jsonify({
            'query': query,
            'results': search_results,
            'personalized': use_personalization,
            'user_id': user_id,
            'total_results': len(search_results)
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/click', methods=['POST'])
def handle_click():
    """Handle result click for feedback learning."""
    data = request.get_json()
    query = data.get('query')
    doc_id = data.get('doc_id')
    user_id = session.get('user_id')
    
    if query and doc_id and user_id:
        try:
            user_profile = get_user_profile(user_id)
            
            # Update click feedback
            retriever = ContextAwareRetriever(doc_index, user_profile)
            retriever.update_user_feedback(query, doc_id, True)
            
            # Add to user's search history if not already there
            # (with clicked document)
            recent_entry = None
            if user_profile.search_history:
                recent_entry = user_profile.search_history[-1]
            
            if not recent_entry or recent_entry.query != query:
                doc = doc_index.get_document(doc_id)
                categories = doc.categories if doc else []
                user_profile.add_search(query, [doc_id], categories)
            else:
                # Add clicked doc to existing entry
                if doc_id not in recent_entry.clicked_docs:
                    recent_entry.clicked_docs.append(doc_id)

            # Persist profile after click update
            try:
                profile_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_profiles')
                os.makedirs(profile_dir, exist_ok=True)
                profile_path = os.path.join(profile_dir, f"{user_profile.user_id}.json")
                user_profile.save_to_file(profile_path)
            except Exception as e:
                print(f"Failed to save user profile after click: {e}")

            return jsonify({'status': 'success'})
            
        except Exception as e:
            print(f"Click handling error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Missing data'}), 400


@app.route('/profile')
def user_profile_page():
    """Display user profile and statistics."""
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('home'))
    
    user_profile = get_user_profile(user_id)
    profile_stats = user_profile.get_statistics()
    interest_profile = user_profile.get_interest_profile()
    
    return render_template('profile.html',
                         user_id=user_id,
                         profile_stats=profile_stats,
                         interest_profile=interest_profile)


@app.route('/compare')
def comparison_page():
    """Page for comparing baseline vs personalized results."""
    return render_template('compare.html')


@app.route('/compare_search', methods=['POST'])
def compare_search():
    """Compare baseline vs personalized search results."""
    if not doc_index:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        user_id = session.get('user_id')
        user_profile = get_user_profile(user_id)
        
        # Baseline search (no personalization)
        baseline_retriever = ContextAwareRetriever(doc_index, None)
        baseline_results = baseline_retriever.search(query, max_results=10, use_context=False)
        
        # Personalized search
        personalized_retriever = ContextAwareRetriever(doc_index, user_profile)
        personalized_results = personalized_retriever.search(query, max_results=10, use_context=True)
        
        # Convert to JSON
        baseline_json = [r.to_dict() for r in baseline_results]
        personalized_json = [r.to_dict() for r in personalized_results]
        
        return jsonify({
            'query': query,
            'baseline_results': baseline_json,
            'personalized_results': personalized_json,
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Comparison error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recommendations')
def recommendations():
    """Get personalized document recommendations."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'No user session'}), 400
    
    try:
        user_profile = get_user_profile(user_id)
        retriever = ContextAwareRetriever(doc_index, user_profile)
        
        recommendations = retriever.get_recommendations(max_results=8)
        recommendations_json = [r.to_dict() for r in recommendations]
        
        return jsonify({
            'recommendations': recommendations_json,
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Recommendations error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/statistics')
def system_statistics():
    """Get system statistics and metrics."""
    try:
        stats = {
            'total_documents': doc_index.get_document_count() if doc_index else 0,
            'total_users': len(user_profiles),
            'categories': doc_index.get_all_categories() if doc_index else [],
            'active_user': session.get('user_id')
        }
        
        if doc_index:
            stats.update(doc_index.get_statistics())
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/reset_profile')
def reset_profile():
    """Reset user profile (for testing)."""
    user_id = session.get('user_id')
    if user_id and user_id in user_profiles:
        del user_profiles[user_id]
    
    return jsonify({'status': 'Profile reset', 'user_id': user_id})


@app.route('/simulate_user/<user_type>')
def simulate_user(user_type):
    """Simulate different user types for demonstration."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'No user session'}), 400
    
    user_profile = get_user_profile(user_id)
    
    # Clear existing history
    user_profile.search_history = []
    user_profile.interest_categories.clear()
    user_profile.keyword_preferences.clear()
    user_profile.topic_preferences.clear()
    
    # Simulate different user types
    if user_type == 'tech':
        # Tech-oriented user
        tech_queries = [
            "machine learning algorithms",
            "python programming tutorial", 
            "database optimization",
            "web development frameworks",
            "artificial intelligence applications",
            "data structures implementation"
        ]
        for query in tech_queries:
            user_profile.add_search(query, [], ['technology', 'programming'])
            
    elif user_type == 'biology':
        # Biology-oriented user
        bio_queries = [
            "animal behavior research",
            "ecosystem conservation methods",
            "reptile species classification",
            "marine biology studies",
            "genetic diversity analysis",
            "environmental protection strategies"
        ]
        for query in bio_queries:
            user_profile.add_search(query, [], ['biology', 'environment'])
            
    elif user_type == 'history':
        # History-oriented user  
        history_queries = [
            "ancient civilizations development",
            "world war timeline",
            "renaissance art movements", 
            "historical archaeological discoveries",
            "cultural heritage preservation",
            "medieval society structure"
        ]
        for query in history_queries:
            user_profile.add_search(query, [], ['history', 'culture'])
    
    return jsonify({
        'status': f'Simulated {user_type} user',
        'user_id': user_id,
        'search_count': len(user_profile.search_history)
    })


if __name__ == '__main__':
    # Initialize the system
    print("Initializing IR system...")
    initialize_system()
    
    # Run the Flask app (development)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    port = int(os.environ.get('PORT', 5000))
    print("Starting web server (development)...")
    print(f"Access the application at: http://localhost:{port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


# Initialize system at import time so WSGI servers (gunicorn) have the index ready.
# Set SKIP_INITIALIZE=1 to avoid heavy initialization during some CI tasks.
if os.environ.get('SKIP_INITIALIZE', '0') != '1':
    try:
        print('Initializing IR system...')
        initialize_system()
    except Exception as e:
        # Print the error but do not crash import. This helps in environments
        # where model downloads or heavy GPU requirements are not available.
        print(f'Warning: system initialization failed at import: {e}')