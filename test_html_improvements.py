"""
Test script to demonstrate HTML Agent improvements with document integration.
"""

from dotenv import load_dotenv
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ContextStore.cs import ContextStore
from Agents.ContentAgent.html_ag import HtmlAgent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_html_agent_with_documents():
    """Test the improved HTML agent with document context."""
    print("🧪 Testing HTML Agent with Document Integration")
    print("=" * 50)
    
    # Initialize context store
    store = ContextStore("test_conversation", None, None)
    
    # Add a sample document to the store
    sample_doc_content = """
    Product Information:
    - Name: EcoSmart Water Bottle
    - Price: $24.99
    - Features: BPA-free, 500ml capacity, temperature retention for 12 hours
    - Materials: Stainless steel with silicone grip
    - Colors: Blue, Green, Silver, Black
    - Warranty: 2 years
    """
    
    doc_data = sample_doc_content.encode('utf-8')
    store.add_document_data_to_context(doc_data, "EcoSmart Water Bottle Product Info")
    
    # Initialize model and HTML agent
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-exp")
    html_agent = HtmlAgent(model=model, store=store)
    
    # Test HTML generation with document context
    description = "Create a product description page for the water bottle using the available product information"
    style_guidelines = "Use professional styling with headings, bullet points, and a clean layout. Use Arial font and appropriate colors."
    
    print("📝 Generating HTML with document context...")
    print(f"Description: {description}")
    print(f"Style Guidelines: {style_guidelines}")
    print()
    
    try:
        result = html_agent.run(description, style_guidelines)
        
        print("✅ HTML Generation Result:")
        print("-" * 30)
        print(f"Final HTML: {result.get('html', 'No HTML generated')}")
        print(f"Score: {result.get('evaluator_score', 'No score')}")
        print(f"Feedback: {result.get('evaluator_feedback', 'No feedback')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_context_generation():
    """Test the context generation function."""
    print("\n🧪 Testing Context Generation Function")
    print("=" * 50)
    
    # Initialize context store with sample data
    store = ContextStore("test_conversation", None, None)
    
    # Add sample document
    doc_content = "This is a sample document about AI trends in 2024."
    store.add_document_data_to_context(doc_content.encode('utf-8'), "AI Trends 2024")
    
    # Add sample image
    try:
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        store.add_image_url_to_context(img_url, "Sample nature image")
    except Exception as e:
        print(f"Note: Could not add image - {e}")
    
    # Initialize HTML agent
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-exp")
    html_agent = HtmlAgent(model=model, store=store)
    
    # Generate context
    context = html_agent.get_context(
        description="Create a blog post about technology",
        style_guidelines="Modern, clean styling",
        previous_context="Previous discussion about AI applications"
    )
    
    print("📋 Generated Context:")
    print("-" * 30)
    print(context)
    
    return True

def main():
    """Main test function."""
    print("🚀 HTML Agent Improvements Test Suite")
    print("=" * 60)
    
    # Test 1: Context generation
    test_context_generation()
    
    # Test 2: HTML generation with documents
    test_html_agent_with_documents()
    
    print("\n🎉 Test suite completed!")
    print("✨ Key improvements demonstrated:")
    print("  - Better context management with get_context() function")
    print("  - Document integration from DocManager")
    print("  - Cleaner prompt structure")
    print("  - Improved HTML generation reliability")

if __name__ == "__main__":
    main() 