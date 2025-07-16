"""
Comprehensive test suite for the Image Agent functionality.
Tests all image processing, management, and AI-powered analysis features.
"""

from dotenv import load_dotenv
import os
import sys
import time
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.store import ContextStore
from agents.sub_agents.image import ImageAgent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class ImageAgentTester:
    def __init__(self):
        """Initialize the image agent tester."""
        self.image_agent = None
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
        
        if isinstance(result, dict) and 'messages' in result:
            # Extract the final response from agent
            final_message = result['messages'][-1].content if result['messages'] else "No response"
            print(f"🤖 Agent Response: {final_message}")
        else:
            print(f"📄 Result: {result}")

    def wait_for_user(self):
        """Wait for user input before continuing."""
        input("\n⏸️  Press Enter to continue to next test...")

    def initialize_image_agent(self):
        """Initialize the image agent for testing."""
        print("🚀 Initializing Image Agent Test Suite...")
        
        # Create context store
        self.store = ContextStore("image_test_conversation", None, None)
        
        # Initialize model
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06")
        
        # Initialize image agent
        self.image_agent = ImageAgent(
            model=model,
            store=self.store,
            checkpoint_path="data/checkpoint.sqlite"
        )
        
        print("✅ Image Agent initialized successfully!")
        print(f"📊 Available tools: {len(self.image_agent.defined_tools)}")
        
        return True

    def test_basic_image_operations(self):
        """Test basic image management operations, including type field."""
        self.print_test_header("BASIC IMAGE OPERATIONS")
        
        tests = [
            ("Get comprehensive statistics about all images in the store", "Initial store statistics"),
            ("Add the image from https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg to the store with caption 'Beautiful nature boardwalk in Wisconsin for testing' and type 'jpg'", "Adding image from URL with type"),
            ("Get all images from the store and show their details", "Retrieving all images from store"),
            ("Search for images with 'nature' in their captions", "Searching images by caption")
        ]
        
        for prompt, description in tests:
            self.print_test_result(
                self.image_agent.run(prompt),
                description
            )
            self.wait_for_user()

    def test_image_analysis(self):
        """Test AI-powered image analysis features."""
        self.print_test_header("AI-POWERED IMAGE ANALYSIS")
        
        # Get image ID for analysis tests
        print("🔍 Getting image ID for analysis tests...")
        images_result = self.image_agent.run("Get all images from the store")
        print(f"Images in store: {images_result}")
        
        print("\n📝 Please check the above output and enter an image ID for testing:")
        image_id = input("Enter image ID: ").strip()
        
        if image_id:
            analysis_tests = [
                (f"Generate an AI-powered detailed caption for image ID {image_id}", "AI Caption Generation"),
                (f"Analyze the content of image ID {image_id} with analysis type 'general'", "General Content Analysis"),
                (f"Analyze the content of image ID {image_id} with analysis type 'objects'", "Object Detection Analysis"),
                (f"Extract all text from image ID {image_id}", "Text Extraction (OCR)"),
                (f"Get comprehensive metadata for image ID {image_id}", "Metadata Extraction")
            ]
            
            for prompt, description in analysis_tests:
                self.print_test_result(
                    self.image_agent.run(prompt),
                    description
                )
                self.wait_for_user()
        else:
            print("⚠️ Skipping analysis tests - no image ID provided")

    def test_image_processing(self):
        """Test image processing capabilities."""
        self.print_test_header("IMAGE PROCESSING")
        
        # Get image ID for processing
        print("🔍 Getting image ID for processing tests...")
        images_result = self.image_agent.run("Get all images from the store")
        print(f"Images in store: {images_result}")
        
        print("\n📝 Please enter an image ID for processing tests:")
        image_id = input("Enter image ID: ").strip()
        
        if image_id:
            processing_tests = [
                (f"Resize image ID {image_id} to 800x600 pixels while maintaining aspect ratio", "Image Resizing"),
                (f"Apply a sharpen filter to image ID {image_id}", "Applying Image Filter"),
                (f"Enhance image ID {image_id} with enhancement type 'brightness' and factor 1.2", "Image Enhancement"),
                (f"Create a thumbnail for image ID {image_id} with max size 150 pixels", "Thumbnail Creation"),
                (f"Convert image ID {image_id} to PNG format", "Format Conversion")
            ]
            
            for prompt, description in processing_tests:
                self.print_test_result(
                    self.image_agent.run(prompt),
                    description
                )
                self.wait_for_user()
        else:
            print("⚠️ Skipping processing tests - no image ID provided")

    def test_advanced_operations(self):
        """Test advanced image operations."""
        self.print_test_header("ADVANCED OPERATIONS")
        
        # Test URL validation
        test_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png",
            "https://invalid-url-example.com/nonexistent.jpg",
            "not-a-url-at-all"
        ]
        
        for url in test_urls:
            self.print_test_result(
                self.image_agent.run(f"Validate if this URL is accessible and points to a valid image: {url}"),
                f"URL Validation for: {url}"
            )
            time.sleep(1)  # Brief pause between requests
        
        self.wait_for_user()
        
        # Get updated statistics
        self.print_test_result(
            self.image_agent.run("Get comprehensive statistics about all images in the store"),
            "Final Store Statistics"
        )
        self.wait_for_user()
        
        # Test image comparison if we have multiple images
        images_result = self.image_agent.run("Get all images from the store")
        print(f"\n📊 Current images: {images_result}")
        
        print("\n🔄 If you have multiple images, enter two image IDs to compare:")
        id1 = input("First image ID (or press Enter to skip): ").strip()
        id2 = input("Second image ID (or press Enter to skip): ").strip()
        
        if id1 and id2:
            self.print_test_result(
                self.image_agent.run(f"Compare images with IDs {id1} and {id2}"),
                "Image Comparison"
            )
        else:
            print("⚠️ Skipping image comparison - insufficient image IDs")

    def test_error_handling(self):
        """Test error handling and edge cases."""
        self.print_test_header("ERROR HANDLING & EDGE CASES")
        
        error_tests = [
            ("Get details for image ID 'nonexistent-id-12345'", "Invalid Image ID Handling"),
            ("Add image from URL 'not-a-valid-url' with caption 'test'", "Invalid URL Handling")
        ]
        
        for prompt, description in error_tests:
            self.print_test_result(
                self.image_agent.run(prompt),
                description
            )
            self.wait_for_user()
        
        # Test invalid filter type
        images_result = self.image_agent.run("Get all images from the store")
        print(f"Images: {images_result}")
        
        image_id = input("\nEnter an image ID for error testing (or press Enter to use 'test-id'): ").strip()
        if not image_id:
            image_id = "test-id"
        
        self.print_test_result(
            self.image_agent.run(f"Apply an 'invalid-filter-type' filter to image ID {image_id}"),
            "Invalid Filter Type Handling"
        )
        self.wait_for_user()

    def run_interactive_mode(self):
        """Run interactive mode for manual testing."""
        self.print_test_header("INTERACTIVE MODE")
        print("🎮 Interactive testing mode - try any image operation!")
        print("Examples:")
        print("  - 'Add image from https://example.com/image.jpg'")
        print("  - 'Generate caption for image ID abc123'")
        print("  - 'Resize all images to 500x500'")
        print("  - Type 'quit' to exit")
        
        while True:
            try:
                prompt = input("\n🎯 Enter command: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print(f"\n🔄 Processing: {prompt}")
                result = self.image_agent.run(prompt)
                self.print_test_result(result, "Interactive Command Result")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 IMAGE AGENT COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        try:
            # Initialize
            if not self.initialize_image_agent():
                return False
            
            # Run test suites
            self.test_basic_image_operations()
            self.test_image_analysis()
            self.test_image_processing()
            self.test_advanced_operations()
            self.test_error_handling()
            
            # Interactive mode
            self.run_interactive_mode()
            
            print("\n🎉 Image Agent test suite completed!")
            print("✅ All major functionality has been tested.")
            return True
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
            print("Please check your environment variables and dependencies.")
            return False


def main():
    """Main function to run image agent tests."""
    tester = ImageAgentTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 