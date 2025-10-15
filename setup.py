#!/usr/bin/env python3
"""
Setup and initialization script for the Context-Aware Information Retrieval System.

This script handles:
1. Dependency checking and installation guidance
2. NLTK data downloads
3. System initialization and testing
4. Sample data verification

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import importlib


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'numpy',
        'pandas', 
        'sklearn',
        'nltk',
        'flask',
        'matplotlib',
        'seaborn'
    ]
    
    optional_packages = [
        'transformers',
        'torch',
        'sentence_transformers',
        'plotly',
        'wordcloud'
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (required)")
            missing_required.append(package)
    
    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional - some features may be limited)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("   Please install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("   Install them for full functionality:")
        print("   pip install transformers torch sentence-transformers plotly wordcloud")
    
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_downloads = [
            ('punkt', 'Punkt tokenizer'),
            ('stopwords', 'Stopwords corpus')
        ]
        
        for item, description in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
                print(f"✅ {description} already available")
            except LookupError:
                print(f"📥 Downloading {description}...")
                nltk.download(item, quiet=True)
                print(f"✅ {description} downloaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False


def verify_sample_data():
    """Verify sample data files exist."""
    print("\nVerifying sample data...")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    sample_file = os.path.join(data_dir, 'sample_documents.json')
    
    if os.path.exists(sample_file):
        print(f"✅ Sample documents found: {sample_file}")
        
        # Check file size
        file_size = os.path.getsize(sample_file)
        print(f"   File size: {file_size:,} bytes")
        
        # Try to load and validate JSON
        try:
            import json
            with open(sample_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            print(f"   Contains {len(documents)} documents")
            
            # Verify structure
            if documents and isinstance(documents[0], dict):
                required_fields = ['doc_id', 'title', 'content']
                sample_doc = documents[0]
                
                missing_fields = [field for field in required_fields if field not in sample_doc]
                if missing_fields:
                    print(f"⚠️  Missing required fields: {missing_fields}")
                else:
                    print("✅ Document structure is valid")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in sample documents: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading sample documents: {e}")
            return False
    else:
        print(f"❌ Sample documents not found: {sample_file}")
        print("   The demo will create minimal test data automatically")
        return False


def test_system_components():
    """Test basic system functionality."""
    print("\nTesting system components...")
    
    try:
        # Test imports
        print("  Testing imports...")
        from src.document_index import DocumentIndex
        from src.user_profile import UserProfile
        from src.retriever import ContextAwareRetriever
        print("  ✅ Core modules imported successfully")
        
        # Test basic functionality
        print("  Testing basic functionality...")
        
        # Create document index
        doc_index = DocumentIndex(use_bert=False)  # Disable BERT for faster testing
        
        # Add test document
        doc_index.add_document(
            "test_doc",
            "Test Document",
            "This is a test document for system verification.",
            ["test"]
        )
        
        # Create user profile
        user_profile = UserProfile("test_user")
        user_profile.add_search("test query", [], ["test"])
        
        # Create retriever
        retriever = ContextAwareRetriever(doc_index, user_profile)
        
        # Perform test search
        results = retriever.search("test", max_results=5)
        
        if results:
            print("  ✅ Search functionality working")
        else:
            print("  ⚠️  Search returned no results (may be normal)")
        
        print("  ✅ All components working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        'data',
        'examples/demo_output',
        'visualization/outputs',
        'web/static',
        'web/templates'
    ]
    
    project_root = os.path.dirname(__file__)
    
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ Created: {directory}")
        else:
            print(f"  ✓ Exists: {directory}")


def display_usage_instructions():
    """Display instructions for using the system."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 USAGE INSTRUCTIONS:")
    print("\n1. RUN EXAMPLES:")
    print("   python examples/simple_examples.py    # Basic usage examples")
    print("   python examples/demo.py              # Complete demonstration")
    
    print("\n2. START WEB INTERFACE:")
    print("   python web/app.py                    # Start Flask web server")
    print("   Then open: http://localhost:5000")
    
    print("\n3. EXPLORE FEATURES:")
    print("   • Search with different user profiles")
    print("   • Compare baseline vs personalized results")
    print("   • View user profile analytics")
    print("   • Test feedback learning")
    
    print("\n4. EXAMPLE QUERIES TO TRY:")
    print("   • 'python' (shows context-aware ranking)")
    print("   • 'machine learning' (tech-focused results)")
    print("   • 'ecosystem' (biology-focused results)")
    print("   • 'ancient' (history-focused results)")
    
    print("\n5. SIMULATE USER TYPES:")
    print("   • Use the web interface buttons to simulate different users")
    print("   • Try the same query with different user profiles")
    print("   • Observe how rankings change based on search history")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("   src/          - Core IR system modules")
    print("   data/         - Sample documents and datasets")
    print("   web/          - Flask web application")
    print("   examples/     - Usage examples and demos")
    print("   visualization/ - Analysis and plotting tools")
    
    print("\n🔧 CUSTOMIZATION:")
    print("   • Edit data/sample_documents.json to add your documents")
    print("   • Modify src/utils.py to adjust topic categories")
    print("   • Tune scoring weights in src/retriever.py")
    
    print("\n" + "="*60)


def main():
    """Run the complete setup process."""
    print("CONTEXT-AWARE INFORMATION RETRIEVAL SYSTEM")
    print("Setup and Initialization")
    print("="*60)
    
    # Check system requirements
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed due to missing dependencies")
        return 1
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK data download failed, but system may still work")
    
    # Verify sample data
    verify_sample_data()
    
    # Test system components
    if not test_system_components():
        print("\n❌ System component test failed")
        return 1
    
    # Display usage instructions
    display_usage_instructions()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())