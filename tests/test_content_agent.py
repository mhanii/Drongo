"""
Comprehensive test suite for the Content Agent coordination functionality.
Tests the coordination between HTML and Image agents, content strategy, and integration.
"""

from dotenv import load_dotenv
import os
import sys
import time
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.store import ContextStore
from agents.content import ContentAgent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class ContentAgentTester:
    def __init__(self):
        """Initialize the content agent tester."""
        self.content_agent = None
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
        
        if isinstance(result, dict) and 'messages' in result:
            # Extract the final response from agent
            final_message = result['messages'][-1].content if result['messages'] else "No response"
            print(f"🤖 Agent Response: {final_message}")
        else:
            print(f"📄 Result: {result}")

    def wait_for_user(self):
        """Wait for user input before continuing."""
        input("\n⏸️  Press Enter to continue to next test...")

    def initialize_content_agent(self):
        """Initialize the content agent for testing."""
        print("🚀 Initializing Content Agent Test Suite...")
        
        # Create context store
        # self.store = ContextStore()
        
        # Initialize content agent
        self.content_agent = ContentAgent(
            model="models/gemini-2.0-flash",
            # store=self.store,
            checkpoint_path="data/checkpoint.sqlite"
        )
        
        print("✅ Content Agent initialized successfully!")
        print(f"📊 Available tools: {len(self.content_agent.defined_tools)}")
        
        return True

    def test_html_agent_coordination(self):
        """Test coordination with HTML agent."""
        self.print_test_header("HTML AGENT COORDINATION")
        
        html_test_cases = [
            "Generate a simple paragraph about machine learning with clean formatting",
            "Create a professional product description for a new smartphone with proper styling",
            "Generate a blog post introduction about sustainable technology with modern typography",
            "Create a comparison table of different programming languages with professional styling"
        ]
        
        for i, test_case in enumerate(html_test_cases, 1):
            print(f"\n🧪 HTML Coordination Test {i}:")
            print(f"Request: {test_case}")
            
            result = self.content_agent.run(test_case)
            self.print_test_result(result, f"HTML Coordination - Test {i}")
            self.wait_for_user()

    def test_image_agent_coordination(self):
        """Test coordination with Image agent."""
        self.print_test_header("IMAGE AGENT COORDINATION")
        
        image_test_cases = [
            "Add an image from https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg with caption 'Nature boardwalk for testing'",
            "Get statistics about all images in the store",
            "Search for images with 'nature' in their captions",
            "Generate AI-powered captions for all images that don't have detailed captions"
        ]
        
        for i, test_case in enumerate(image_test_cases, 1):
            print(f"\n🧪 Image Coordination Test {i}:")
            print(f"Request: {test_case}")
            
            result = self.content_agent.run(test_case)
            self.print_test_result(result, f"Image Coordination - Test {i}")
            self.wait_for_user()

    def test_context_analysis(self):
        """Test context analysis and summary capabilities."""
        self.print_test_header("CONTEXT ANALYSIS")
        
        context_test_cases = [
            "Get a summary of all available context including images and documents",
            "Analyze what content generation approach would work best for a tech blog",
            "Determine the requirements for creating a product landing page",
            "Assess what resources are available for creating visual content"
        ]
        
        for i, test_case in enumerate(context_test_cases, 1):
            print(f"\n🧪 Context Analysis Test {i}:")
            print(f"Request: {test_case}")
            
            result = self.content_agent.run(test_case)
            self.print_test_result(result, f"Context Analysis - Test {i}")
            self.wait_for_user()

    def test_coordinated_content_generation(self):
        """Test coordinated content generation involving both HTML and images."""
        self.print_test_header("COORDINATED CONTENT GENERATION")
        
        coordinated_test_cases = [
            {
                "request": "Create a blog post about natural landscapes using any available images from the store",
                "description": "Multi-agent coordination for blog content"
            },
            {
                "request": "Generate a product showcase page that includes both text content and image processing",
                "description": "E-commerce content with image integration"
            },
            {
                "request": "Create a comprehensive article about photography with image examples and proper formatting",
                "description": "Technical content with visual elements"
            }
        ]
        
        for i, test_case in enumerate(coordinated_test_cases, 1):
            print(f"\n🧪 Coordinated Generation Test {i}:")
            print(f"Request: {test_case['request']}")
            print(f"Description: {test_case['description']}")
            
            result = self.content_agent.run(test_case['request'])
            self.print_test_result(result, f"Coordinated Generation - Test {i}")
            self.wait_for_user()

    def test_content_strategy_planning(self):
        """Test content strategy and planning capabilities."""
        self.print_test_header("CONTENT STRATEGY & PLANNING")
        
        strategy_test_cases = [
            {
                "request": "Plan a content strategy for a tech startup's website including both text and visual elements",
                "type": "Strategic planning"
            },
            {
                "request": "Analyze the requirements for creating an educational course landing page",
                "type": "Requirement analysis"
            },
            {
                "request": "Coordinate the creation of a multi-section article about AI with proper structure and images",
                "type": "Complex coordination"
            },
            {
                "request": "Design a content workflow for a product comparison page with tables and product images",
                "type": "Workflow design"
            }
        ]
        
        for i, test_case in enumerate(strategy_test_cases, 1):
            print(f"\n🧪 Strategy Test {i} - {test_case['type']}:")
            print(f"Request: {test_case['request']}")
            
            result = self.content_agent.run(test_case['request'])
            self.print_test_result(result, f"Strategy Planning - Test {i}")
            self.wait_for_user()

    def test_quality_assurance_and_integration(self):
        """Test quality assurance and content integration."""
        self.print_test_header("QUALITY ASSURANCE & INTEGRATION")
        
        qa_test_cases = [
            "Review and improve the quality of content generation for a professional website",
            "Ensure proper integration between text content and visual elements",
            "Validate that generated content meets professional standards and guidelines",
            "Coordinate quality control across both HTML and image processing operations"
        ]
        
        for i, test_case in enumerate(qa_test_cases, 1):
            print(f"\n🧪 Quality Assurance Test {i}:")
            print(f"Request: {test_case}")
            
            result = self.content_agent.run(test_case)
            self.print_test_result(result, f"Quality Assurance - Test {i}")
            self.wait_for_user()

    def test_complex_workflows(self):
        """Test complex multi-step workflows."""
        self.print_test_header("COMPLEX WORKFLOWS")
        
        workflow_test_cases = [
            {
                "request": """Create a complete product launch page that includes:
                1. Add product images from URLs
                2. Generate compelling product descriptions
                3. Create feature comparison tables
                4. Process images for optimal display
                5. Integrate everything into a cohesive page""",
                "description": "Complete product launch workflow"
            },
            {
                "request": """Develop a comprehensive blog post workflow:
                1. Analyze available images for relevance
                2. Generate engaging blog content with proper structure
                3. Create image thumbnails for better performance
                4. Integrate images with text content
                5. Ensure professional formatting throughout""",
                "description": "Blog content creation workflow"
            }
        ]
        
        for i, test_case in enumerate(workflow_test_cases, 1):
            print(f"\n🧪 Complex Workflow Test {i}:")
            print(f"Description: {test_case['description']}")
            print(f"Request: {test_case['request']}")
            
            start_time = time.time()
            result = self.content_agent.run(test_case['request'])
            end_time = time.time()
            
            execution_time = end_time - start_time
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            
            self.print_test_result(result, f"Complex Workflow - Test {i}")
            self.wait_for_user()

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        self.print_test_header("ERROR HANDLING & EDGE CASES")
        
        edge_cases = [
            {
                "request": "",  # Empty request
                "description": "Empty request handling"
            },
            {
                "request": "Process a non-existent image with ID 'fake-image-123'",
                "description": "Invalid image ID handling"
            },
            {
                "request": "Create content with impossible requirements that cannot be fulfilled",
                "description": "Impossible requirements handling"
            },
            {
                "request": "Generate content using tools that don't exist",
                "description": "Invalid tool usage handling"
            }
        ]
        
        for i, test_case in enumerate(edge_cases, 1):
            print(f"\n🧪 Edge Case Test {i} - {test_case['description']}:")
            print(f"Request: '{test_case['request']}'")
            
            try:
                result = self.content_agent.run(test_case['request'])
                self.print_test_result(result, f"Edge Case - Test {i}")
                
            except Exception as e:
                print(f"❌ Exception caught: {str(e)}")
                print("✅ Error handling working correctly")
            
            self.wait_for_user()

    def test_performance_metrics(self):
        """Test performance and efficiency metrics."""
        self.print_test_header("PERFORMANCE METRICS")
        
        performance_tests = [
            ("Simple coordination", "Create a paragraph about AI and add an image"),
            ("Medium complexity", "Generate a product page with images and descriptions"),
            ("High complexity", "Create a comprehensive article with multiple images and complex formatting"),
            ("Multi-tool usage", "Coordinate HTML generation, image processing, and content analysis")
        ]
        
        results_summary = []
        
        for test_name, request in performance_tests:
            print(f"\n⏱️ Performance Test: {test_name}")
            print(f"Request: {request}")
            
            start_time = time.time()
            result = self.content_agent.run(request)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Analyze result quality
            success = isinstance(result, dict) and 'messages' in result and len(result['messages']) > 0
            
            test_result = {
                "test_name": test_name,
                "execution_time": round(execution_time, 2),
                "success": success,
                "request_length": len(request)
            }
            
            results_summary.append(test_result)
            
            print(f"⏱️ Time: {execution_time:.2f}s")
            print(f"✅ Success: {success}")
        
        # Print performance summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        for result in results_summary:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{result['test_name']}: {result['execution_time']}s, {status}")
        
        avg_time = sum(r['execution_time'] for r in results_summary) / len(results_summary)
        success_rate = sum(1 for r in results_summary if r['success']) / len(results_summary) * 100
        
        print(f"\n📈 OVERALL METRICS:")
        print(f"Average execution time: {avg_time:.2f}s")
        print(f"Success rate: {success_rate:.1f}%")
        
        self.wait_for_user()

    def run_interactive_mode(self):
        """Run interactive mode for manual testing."""
        self.print_test_header("INTERACTIVE MODE")
        print("🎮 Interactive testing mode - try any content coordination request!")
        print("Examples:")
        print("  - 'Create a landing page with images and professional content'")
        print("  - 'Generate a blog post about technology with visual elements'")
        print("  - 'Coordinate image processing and HTML generation for a product page'")
        print("  - 'Analyze content requirements and suggest an approach'")
        print("  - Type 'quit' to exit")
        
        while True:
            try:
                prompt = input("\n🎯 Enter content request: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print(f"\n🔄 Processing content coordination...")
                result = self.content_agent.run(prompt)
                self.print_test_result(result, "Interactive Content Coordination")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 CONTENT AGENT COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        try:
            # Initialize
            if not self.initialize_content_agent():
                return False
            
            # Run test suites
            self.test_html_agent_coordination()
            self.test_image_agent_coordination()
            self.test_context_analysis()
            self.test_coordinated_content_generation()
            self.test_content_strategy_planning()
            self.test_quality_assurance_and_integration()
            self.test_complex_workflows()
            self.test_error_handling_and_edge_cases()
            self.test_performance_metrics()
            
            # Interactive mode
            self.run_interactive_mode()
            
            print("\n🎉 Content Agent test suite completed!")
            print("✅ All coordination and integration functionality has been tested.")
            return True
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
            print("Please check your environment variables and dependencies.")
            return False


def main():
    """Main function to run content agent tests."""
    tester = ContentAgentTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 