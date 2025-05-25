"""
Content Agent System - Main Entry Point
This file provides a simple interface to access the comprehensive test suites.
"""

from dotenv import load_dotenv
import os

load_dotenv()

def print_welcome():
    """Print welcome message and instructions."""
    print("🚀" + "="*58 + "🚀")
    print("🎉        WELCOME TO CONTENT AGENT SYSTEM v2.0        🎉")
    print("🚀" + "="*58 + "🚀")
    print()
    print("✨ SYSTEM CAPABILITIES:")
    print("   🖼️  Advanced Image Processing & AI Analysis")
    print("   📝 Sophisticated HTML Generation with Validation")
    print("   🤖 Intelligent Content Coordination & Strategy")
    print("   📊 Comprehensive Quality Assurance & Integration")
    print()
    print("🧪 TESTING & VALIDATION:")
    print("   The system includes comprehensive test suites to validate")
    print("   all functionality and ensure everything works correctly.")
    print()

def print_usage_instructions():
    """Print usage instructions."""
    print("📋 HOW TO USE THE SYSTEM:")
    print("-" * 40)
    print()
    print("🧪 1. RUN COMPREHENSIVE TESTS:")
    print("   cd tests")
    print("   python run_tests.py")
    print("   (Interactive menu with all test options)")
    print()
    print("🎯 2. RUN SPECIFIC TESTS:")
    print("   python tests/run_tests.py img      # Image Agent tests")
    print("   python tests/run_tests.py html     # HTML Agent tests") 
    print("   python tests/run_tests.py content  # Content Agent tests")
    print("   python tests/run_tests.py all      # All tests sequential")
    print("   python tests/run_tests.py summary  # All tests with summaries")
    print()
    print("🔧 3. DIRECT TEST EXECUTION:")
    print("   python tests/test_img_agent.py     # Image Agent only")
    print("   python tests/test_html_agent.py    # HTML Agent only")
    print("   python tests/test_content_agent.py # Content Agent only")
    print()
    print("📚 4. SYSTEM DOCUMENTATION:")
    print("   See IMPROVEMENTS_SUMMARY.md for detailed system overview")
    print("   Each test file contains comprehensive functionality tests")
    print()

def print_system_status():
    """Print system status and requirements."""
    print("⚙️  SYSTEM STATUS:")
    print("-" * 30)
    
    # Check environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("   ✅ Google API Key: Configured")
    else:
        print("   ❌ Google API Key: Missing (required for AI features)")
    
    # Check database files
    db_files = ["checkpoint.sqlite", "image.sqlite", "doc.sqlite"]
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"   ✅ Database {db_file}: Available")
        else:
            print(f"   ⚠️  Database {db_file}: Will be created on first use")
    
    # Check test files
    test_files = [
        "tests/test_img_agent.py",
        "tests/test_html_agent.py", 
        "tests/test_content_agent.py",
        "tests/run_tests.py"
    ]
    
    all_tests_available = all(os.path.exists(f) for f in test_files)
    if all_tests_available:
        print("   ✅ Test Suites: All available")
    else:
        print("   ❌ Test Suites: Some missing")
    
    print()

def main():
    """Main function providing system overview and usage instructions."""
    print_welcome()
    print_system_status()
    print_usage_instructions()
    
    print("🎯 QUICK START RECOMMENDATION:")
    print("-" * 35)
    print("   1. Run: python tests/run_tests.py")
    print("   2. Select option 1 (Image Agent Tests) to start")
    print("   3. Follow the interactive prompts")
    print("   4. Explore other test suites as needed")
    print()
    
    print("💡 TIPS:")
    print("   • Tests are interactive and guide you through each feature")
    print("   • Each test suite can be run independently")
    print("   • Use the test runner for the best experience")
    print("   • Check IMPROVEMENTS_SUMMARY.md for detailed capabilities")
    print()
    
    # Ask if user wants to run tests immediately
    try:
        choice = input("🚀 Would you like to run the test suite now? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print("\n🔄 Launching test runner...")
            import subprocess
            import sys
            env = os.environ.copy()  # Start with current env
            subprocess.run([sys.executable, "tests/run_tests.py"], env=env)
        else:
            print("👍 Great! Run the tests when you're ready.")
            print("   Command: python tests/run_tests.py")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n⚠️  Note: {str(e)}")
        print("You can manually run: python tests/run_tests.py")

if __name__ == "__main__":
    main()
