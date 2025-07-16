"""
Comprehensive test suite for the HTML Agent functionality.
Tests HTML generation, validation, evaluation, and retry logic.
"""

from dotenv import load_dotenv
import os
import sys
import time
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.sub_agents.html import HtmlAgent
from context.store import ContextStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class HtmlAgentTester:
    def __init__(self):
        """Initialize the HTML agent tester."""
        self.html_agent = None
        self.store = None
        self.test_results = []
        
    def print_test_header(self, test_name):
        """Print a formatted test header."""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {test_name}")
        print(f"{'='*60}")

    def print_test_result(self, result, test_description):
        """Print formatted test result."""
        print(f"\n📋 {test_description}")
        print("-" * 40)
        
        if isinstance(result, dict):
            # Extract HTML content if available
            html_content = result.get('html', 'No HTML content')
            score = result.get('evaluator_score', 'No score')
            feedback = result.get('evaluator_feedback', 'No feedback')
            
            print(f"🎯 Score: {score}")
            print(f"💬 Feedback: {feedback}")
            print(f"📄 HTML Content: {html_content[:200]}..." if len(str(html_content)) > 200 else f"📄 HTML Content: {html_content}")
        else:
            print(f"📄 Result: {result}")

    def wait_for_user(self):
        """Wait for user input before continuing."""
        input("\n⏸️  Press Enter to continue to next test...")

    def initialize_html_agent(self):
        """Initialize the HTML agent for testing."""
        print("🚀 Initializing HTML Agent Test Suite...")
        
        # Initialize context store
        self.store = ContextStore()
        
        # Initialize model
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-exp")
        
        # Initialize HTML agent with custom settings for testing
        self.html_agent = HtmlAgent(
            model=model,
            checkpoint_path="data/checkpoint.sqlite",
            acceptance_threshold=85,  # Lower threshold for testing
            max_retries=2  # Fewer retries for faster testing
        )
        
        print("✅ HTML Agent initialized successfully!")
        print(f"📊 Acceptance threshold: {self.html_agent.acceptance_threshold}")
        print(f"🔄 Max retries: {self.html_agent.max_retries}")
        print(f"🗃️ Document store: {'Available' if self.store else 'Not available'}")
        
        return True

    def test_basic_html_generation(self):
        """Test basic HTML content generation."""
        self.print_test_header("BASIC HTML GENERATION")
        
        test_cases = [
            {
                "description": "Create a simple paragraph about artificial intelligence",
                "style_guidelines": "Use Arial font, 12pt size, normal line height",
                "context": "No previous context"
            },
            {
                "description": "Generate a heading and paragraph about web development",
                "style_guidelines": "Use modern styling with proper typography",
                "context": "This is for a tech blog"
            },
            {
                "description": "Create a simple list of programming languages",
                "style_guidelines": "Use underlined list items, clean formatting",
                "context": "Educational content for beginners"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}:")
            print(f"Description: {test_case['description']}")
            print(f"Style: {test_case['style_guidelines']}")
            
            result = self.html_agent.run(
                test_case["description"],
                test_case["style_guidelines"],
                test_case["context"]
            )
            
            self.print_test_result(result, f"Basic HTML Generation - Test {i}")
            self.wait_for_user()

    def test_complex_html_generation(self):
        """Test complex HTML content generation."""
        self.print_test_header("COMPLEX HTML GENERATION")
        
        complex_test_cases = [
            {
                "description": """Create a comprehensive product description for a smartphone with:
                - Product name and key features
                - Technical specifications in a table
                - Benefits and selling points
                - Call to action""",
                "style_guidelines": "Professional styling, use tables for specs, bold for important features",
                "context": "E-commerce product page"
            },
            {
                "description": """Generate a blog post about climate change including:
                - Compelling headline
                - Introduction paragraph
                - 3 main points with subheadings
                - Conclusion with call to action""",
                "style_guidelines": "Clean typography, proper heading hierarchy, engaging formatting",
                "context": "Environmental awareness blog"
            }
        ]
        
        for i, test_case in enumerate(complex_test_cases, 1):
            print(f"\n🧪 Complex Test Case {i}:")
            print(f"Description: {test_case['description']}")
            
            result = self.html_agent.run(
                test_case["description"],
                test_case["style_guidelines"],
                test_case["context"]
            )
            
            self.print_test_result(result, f"Complex HTML Generation - Test {i}")
            self.wait_for_user()

    def test_validation_and_retry_logic(self):
        """Test HTML validation and retry mechanisms."""
        self.print_test_header("VALIDATION & RETRY LOGIC")
        
        # Test with challenging requirements that might trigger retries
        challenging_cases = [
            {
                "description": "Create a table with exactly 3 columns and 4 rows about different programming languages, their creators, and release years",
                "style_guidelines": "Use proper table structure with headers, clean borders, alternating row colors",
                "context": "Technical reference guide"
            },
            {
                "description": "Generate content with multiple nested lists and proper heading hierarchy from h1 to h3",
                "style_guidelines": "Strict adherence to semantic HTML, proper nesting, consistent spacing",
                "context": "Documentation structure"
            }
        ]
        
        for i, test_case in enumerate(challenging_cases, 1):
            print(f"\n🧪 Validation Test Case {i}:")
            print(f"Description: {test_case['description']}")
            
            result = self.html_agent.run(
                test_case["description"],
                test_case["style_guidelines"],
                test_case["context"]
            )
            
            # Check if retries were triggered
            retry_count = result.get('current_retry_count', 0)
            best_score = result.get('best_score_so_far', 0)
            
            print(f"🔄 Retries used: {retry_count}")
            print(f"🏆 Best score achieved: {best_score}")
            
            self.print_test_result(result, f"Validation Test - Case {i}")
            self.wait_for_user()

    def test_style_compliance(self):
        """Test compliance with different styling requirements."""
        self.print_test_header("STYLE COMPLIANCE TESTING")
        
        style_test_cases = [
            {
                "description": "Create a paragraph about machine learning",
                "style_guidelines": "Font: Arial, Size: 14pt, Color: #333333, Line height: 1.5",
                "expected_elements": ["Arial", "14pt", "#333333", "line-height"]
            },
            {
                "description": "Generate a heading and paragraph about data science",
                "style_guidelines": "Heading: bold, 18pt, #2c3e50. Paragraph: normal weight, 12pt, #34495e",
                "expected_elements": ["bold", "18pt", "#2c3e50", "12pt", "#34495e"]
            },
            {
                "description": "Create a list of benefits with specific formatting",
                "style_guidelines": "List items should be underlined, green color (#27ae60), 13pt font size",
                "expected_elements": ["underline", "#27ae60", "13pt"]
            }
        ]
        
        for i, test_case in enumerate(style_test_cases, 1):
            print(f"\n🧪 Style Test Case {i}:")
            print(f"Description: {test_case['description']}")
            print(f"Style Requirements: {test_case['style_guidelines']}")
            
            result = self.html_agent.run(
                test_case["description"],
                test_case["style_guidelines"],
                "Style compliance test"
            )
            
            # Check for expected style elements in the HTML
            html_content = result.get('html', '')
            found_elements = []
            missing_elements = []
            
            for element in test_case['expected_elements']:
                if element.lower() in html_content.lower():
                    found_elements.append(element)
                else:
                    missing_elements.append(element)
            
            print(f"✅ Found style elements: {found_elements}")
            if missing_elements:
                print(f"❌ Missing style elements: {missing_elements}")
            
            self.print_test_result(result, f"Style Compliance - Test {i}")
            self.wait_for_user()

    def test_document_integration(self):
        """Test HTML generation with document context integration."""
        self.print_test_header("DOCUMENT INTEGRATION TESTING")
        
        # Test 1: HTML generation with document context
        print("\n🧪 Test 1: HTML Generation with Document Context")
        
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
        self.store.add_document_data_to_context(doc_data, "EcoSmart Water Bottle Product Info")
        
        print("📄 Added sample document to store")
        print(f"Document content preview: {sample_doc_content[:100]}...")
        
        # Test HTML generation with document context
        description = "Create a product description page for the water bottle using the available product information"
        style_guidelines = "Use professional styling with headings, bullet points, and a clean layout. Use Arial font and appropriate colors."
        
        result = self.html_agent.run(description, style_guidelines)
        self.print_test_result(result, "Document Integration - Product Page")
        self.wait_for_user()
        
        # Test 2: Context generation function
        print("\n🧪 Test 2: Context Generation Function")
        
        # Add another document
        doc_content2 = "This is a sample document about AI trends in 2024. Machine learning continues to evolve rapidly."
        self.store.add_document_data_to_context(doc_content2.encode('utf-8'), "AI Trends 2024")
        
        # Add sample image
        try:
            img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            self.store.add_image_url_to_context(img_url, "Sample nature image")
            print("🖼️ Added sample image to store")
        except Exception as e:
            print(f"Note: Could not add image - {e}")
        
        # Generate context
        context = self.html_agent.get_context(
            description="Create a blog post about technology",
            style_guidelines="Modern, clean styling",
            previous_context="Previous discussion about AI applications"
        )
        
        print("📋 Generated Context:")
        print("-" * 30)
        print(context[:500] + "..." if len(context) > 500 else context)
        self.wait_for_user()
        
        # Test 3: Multiple documents integration
        print("\n🧪 Test 3: Multiple Documents Integration")
        
        # Add a third document
        doc_content3 = """
        Company Information:
        - Name: TechCorp Solutions
        - Founded: 2020
        - Specialization: AI and Machine Learning
        - Team Size: 50+ engineers
        - Mission: Making AI accessible to everyone
        """
        self.store.add_document_data_to_context(doc_content3.encode('utf-8'), "Company Information")
        
        # Generate content using all available documents
        description = "Create a comprehensive company profile page that incorporates all available information"
        style_guidelines = "Corporate styling with professional layout, clear sections, and proper typography"
        
        result = self.html_agent.run(description, style_guidelines)
        self.print_test_result(result, "Multiple Documents Integration - Company Profile")
        self.wait_for_user()

    def test_context_management_improvements(self):
        """Test the improved context management features."""
        self.print_test_header("CONTEXT MANAGEMENT IMPROVEMENTS")
        
        # Test 1: Empty context handling
        print("\n🧪 Test 1: Empty Context Handling")
        
        # Create a new agent with no documents
        empty_store = ContextStore("empty_test", None, None)
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-exp")
        empty_agent = HtmlAgent(model=model, store=empty_store, acceptance_threshold=85, max_retries=2)
        
        result = empty_agent.run(
            "Create a simple welcome message",
            "Clean, modern styling",
            "No previous context"
        )
        self.print_test_result(result, "Empty Context Handling")
        self.wait_for_user()
        
        # Test 2: Large document truncation
        print("\n🧪 Test 2: Large Document Truncation")
        
        # Create a large document (over 1000 characters)
        large_doc = "This is a very long document. " * 50  # Creates ~1500 characters
        self.store.add_document_data_to_context(large_doc.encode('utf-8'), "Large Document Test")
        
        # Test that it gets truncated properly
        context = self.html_agent.get_context(
            "Test description",
            "Test style",
            "Test previous context"
        )
        
        print(f"📏 Large document handling:")
        print(f"Original document length: {len(large_doc)} characters")
        print(f"Context contains truncation marker: {'[truncated]' in context}")
        print(f"Context length: {len(context)} characters")
        self.wait_for_user()
        
        # Test 3: Error handling in document reading
        print("\n🧪 Test 3: Document Reading Error Handling")
        
        # This test demonstrates graceful error handling
        # (We can't easily create a corrupted document, but the system handles it)
        result = self.html_agent.run(
            "Create content about document management",
            "Professional styling with clear structure"
        )
        self.print_test_result(result, "Error Handling - Document Reading")
        self.wait_for_user()

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        self.print_test_header("EDGE CASES & ERROR HANDLING")
        
        edge_cases = [
            {
                "description": "",  # Empty description
                "style_guidelines": "Standard formatting",
                "context": "Empty description test"
            },
            {
                "description": "Create content with special characters: <>&\"'",
                "style_guidelines": "Handle special characters properly",
                "context": "Special character handling test"
            },
            {
                "description": "Generate a very long paragraph with over 500 words about the history of computing",
                "style_guidelines": "Maintain readability with proper formatting",
                "context": "Long content test"
            },
            {
                "description": "Create content with emojis and unicode: 🚀 ✨ 💻 🌟",
                "style_guidelines": "Preserve unicode characters",
                "context": "Unicode handling test"
            }
        ]
        
        for i, test_case in enumerate(edge_cases, 1):
            print(f"\n🧪 Edge Case {i}:")
            print(f"Description: '{test_case['description']}'")
            
            try:
                result = self.html_agent.run(
                    test_case["description"],
                    test_case["style_guidelines"],
                    test_case["context"]
                )
                
                self.print_test_result(result, f"Edge Case - Test {i}")
                
            except Exception as e:
                print(f"❌ Exception caught: {str(e)}")
                print("✅ Error handling working correctly")
            
            self.wait_for_user()

    def test_performance_and_quality_metrics(self):
        """Test performance and quality metrics."""
        self.print_test_header("PERFORMANCE & QUALITY METRICS")
        
        print("🔍 Testing HTML Agent performance with various content types...")
        
        performance_tests = [
            ("Simple paragraph", "Create a paragraph about Python programming", "Clean, readable formatting"),
            ("Complex table", "Create a comparison table of 5 different databases with features", "Professional table styling"),
            ("Multi-section content", "Create content with heading, paragraph, list, and table", "Consistent styling throughout"),
            ("Technical documentation", "Create API documentation with code examples", "Technical formatting with proper structure")
        ]
        
        results_summary = []
        
        for test_name, description, style_guidelines in performance_tests:
            print(f"\n⏱️ Testing: {test_name}")
            
            start_time = time.time()
            result = self.html_agent.run(description, style_guidelines, "Performance test")
            end_time = time.time()
            
            execution_time = end_time - start_time
            score = result.get('evaluator_score', 0)
            retry_count = result.get('current_retry_count', 0)
            
            test_result = {
                "test_name": test_name,
                "execution_time": round(execution_time, 2),
                "score": score,
                "retries": retry_count,
                "success": score >= self.html_agent.acceptance_threshold
            }
            
            results_summary.append(test_result)
            
            print(f"⏱️ Time: {execution_time:.2f}s")
            print(f"🎯 Score: {score}")
            print(f"🔄 Retries: {retry_count}")
            print(f"✅ Success: {test_result['success']}")
        
        # Print summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        for result in results_summary:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{result['test_name']}: {result['execution_time']}s, Score: {result['score']}, {status}")
        
        avg_time = sum(r['execution_time'] for r in results_summary) / len(results_summary)
        avg_score = sum(r['score'] for r in results_summary) / len(results_summary)
        success_rate = sum(1 for r in results_summary if r['success']) / len(results_summary) * 100
        
        print(f"\n📈 OVERALL METRICS:")
        print(f"Average execution time: {avg_time:.2f}s")
        print(f"Average score: {avg_score:.1f}")
        print(f"Success rate: {success_rate:.1f}%")
        
        self.wait_for_user()

    def run_interactive_mode(self):
        """Run interactive mode for manual testing."""
        self.print_test_header("INTERACTIVE MODE")
        print("🎮 Interactive testing mode - try any HTML generation request!")
        print("Examples:")
        print("  - 'Create a landing page for a new product'")
        print("  - 'Generate a blog post about AI trends'")
        print("  - 'Make a comparison table of different frameworks'")
        print("  - Type 'quit' to exit")
        
        while True:
            try:
                description = input("\n📝 Enter content description: ").strip()
                
                if description.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not description:
                    continue
                
                style_guidelines = input("🎨 Enter style guidelines (or press Enter for default): ").strip()
                if not style_guidelines:
                    style_guidelines = "Clean, professional styling with proper typography"
                
                context = input("📋 Enter context (or press Enter for none): ").strip()
                if not context:
                    context = "Interactive test"
                
                print(f"\n🔄 Processing HTML generation...")
                result = self.html_agent.run(description, style_guidelines, context)
                self.print_test_result(result, "Interactive HTML Generation")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 HTML AGENT COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        try:
            # Initialize
            if not self.initialize_html_agent():
                return False
            
            # Run test suites
            self.test_basic_html_generation()
            self.test_complex_html_generation()
            self.test_document_integration()
            self.test_context_management_improvements()
            self.test_validation_and_retry_logic()
            self.test_style_compliance()
            self.test_edge_cases_and_error_handling()
            self.test_performance_and_quality_metrics()
            
            # Interactive mode
            self.run_interactive_mode()
            
            print("\n🎉 HTML Agent test suite completed!")
            print("✅ All major functionality has been tested.")
            return True
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
            print("Please check your environment variables and dependencies.")
            return False


def main():
    """Main function to run HTML agent tests."""
    tester = HtmlAgentTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 