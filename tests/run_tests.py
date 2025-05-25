"""
Test Runner for Content Agent System
Provides a menu interface to run individual test suites or all tests.
"""

import os
import sys
import importlib.util
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


class TestRunner:
    def __init__(self):
        """Initialize the test runner."""
        self.test_modules = {
            "1": {
                "name": "Image Agent Tests",
                "module": "test_img_agent",
                "description": "Test image processing, AI analysis, and management features"
            },
            "2": {
                "name": "HTML Agent Tests", 
                "module": "test_html_agent",
                "description": "Test HTML generation, document integration, validation, and retry logic"
            },
            "3": {
                "name": "Content Agent Tests",
                "module": "test_content_agent", 
                "description": "Test coordination, strategy, and integration features"
            }
        }

    def print_banner(self):
        """Print the test runner banner."""
        print("🧪" + "="*58 + "🧪")
        print("🚀           CONTENT AGENT SYSTEM TEST RUNNER           🚀")
        print("🧪" + "="*58 + "🧪")
        print()

    def print_menu(self):
        """Print the test selection menu."""
        print("📋 Available Test Suites:")
        print("-" * 40)
        
        for key, test_info in self.test_modules.items():
            print(f"  {key}. {test_info['name']}")
            print(f"     {test_info['description']}")
            print()
        
        print("  4. Run All Tests (Sequential)")
        print("  5. Run All Tests (With Summaries)")
        print("  0. Exit")
        print()

    def load_and_run_test(self, module_name):
        """Load and run a specific test module."""
        try:
            # Import the test module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(os.path.dirname(__file__), f"{module_name}.py")
            )
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Run the main function
            print(f"🚀 Starting {module_name}...")
            test_module.main()
            print(f"✅ Completed {module_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error running {module_name}: {str(e)}")
            return False

    def run_all_tests_sequential(self):
        """Run all tests sequentially."""
        print("🚀 RUNNING ALL TESTS SEQUENTIALLY")
        print("=" * 50)
        
        results = {}
        
        for key, test_info in self.test_modules.items():
            print(f"\n🧪 Starting: {test_info['name']}")
            print("-" * 30)
            
            success = self.load_and_run_test(test_info['module'])
            results[test_info['name']] = success
            
            if success:
                print(f"✅ {test_info['name']} completed successfully")
            else:
                print(f"❌ {test_info['name']} failed")
            
            print("\n" + "="*50)
        
        # Print final summary
        self.print_test_summary(results)

    def run_all_tests_with_summaries(self):
        """Run all tests and provide detailed summaries."""
        print("🚀 RUNNING ALL TESTS WITH DETAILED SUMMARIES")
        print("=" * 50)
        
        results = {}
        detailed_results = {}
        
        for key, test_info in self.test_modules.items():
            print(f"\n🧪 Starting: {test_info['name']}")
            print(f"📝 Description: {test_info['description']}")
            print("-" * 40)
            
            try:
                success = self.load_and_run_test(test_info['module'])
                results[test_info['name']] = success
                detailed_results[test_info['name']] = {
                    "success": success,
                    "description": test_info['description'],
                    "module": test_info['module']
                }
                
            except Exception as e:
                results[test_info['name']] = False
                detailed_results[test_info['name']] = {
                    "success": False,
                    "description": test_info['description'],
                    "module": test_info['module'],
                    "error": str(e)
                }
            
            print("\n" + "="*50)
        
        # Print comprehensive summary
        self.print_comprehensive_summary(detailed_results)

    def print_test_summary(self, results):
        """Print a summary of test results."""
        print("\n🎯 TEST EXECUTION SUMMARY")
        print("=" * 40)
        
        total_tests = len(results)
        passed_tests = sum(1 for success in results.values() if success)
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total Test Suites: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 30)
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        if failed_tests > 0:
            print(f"\n⚠️  {failed_tests} test suite(s) failed. Check the output above for details.")
        else:
            print(f"\n🎉 All test suites passed successfully!")

    def print_comprehensive_summary(self, detailed_results):
        """Print a comprehensive summary with details."""
        print("\n🎯 COMPREHENSIVE TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(detailed_results)
        passed_tests = sum(1 for result in detailed_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 OVERVIEW:")
        print(f"  Total Test Suites: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 40)
        
        for test_name, result in detailed_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"\n🧪 {test_name}: {status}")
            print(f"   📝 {result['description']}")
            print(f"   📁 Module: {result['module']}")
            
            if not result['success'] and 'error' in result:
                print(f"   ❌ Error: {result['error']}")
        
        print(f"\n{'='*50}")
        
        if failed_tests > 0:
            print(f"⚠️  {failed_tests} test suite(s) failed.")
            print("💡 Recommendations:")
            print("   - Check environment variables (.env file)")
            print("   - Verify API keys and model access")
            print("   - Ensure all dependencies are installed")
            print("   - Check database permissions and paths")
        else:
            print("🎉 ALL TEST SUITES PASSED SUCCESSFULLY!")
            print("✨ Your Content Agent System is working perfectly!")

    def run_interactive_menu(self):
        """Run the interactive test selection menu."""
        while True:
            self.print_banner()
            self.print_menu()
            
            try:
                choice = input("🎯 Select a test suite (0-5): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                
                elif choice in self.test_modules:
                    test_info = self.test_modules[choice]
                    print(f"\n🚀 Running: {test_info['name']}")
                    print(f"📝 Description: {test_info['description']}")
                    print("-" * 40)
                    
                    success = self.load_and_run_test(test_info['module'])
                    
                    if success:
                        print(f"\n✅ {test_info['name']} completed successfully!")
                    else:
                        print(f"\n❌ {test_info['name']} encountered errors.")
                    
                    input("\n⏸️  Press Enter to return to menu...")
                
                elif choice == "4":
                    self.run_all_tests_sequential()
                    input("\n⏸️  Press Enter to return to menu...")
                
                elif choice == "5":
                    self.run_all_tests_with_summaries()
                    input("\n⏸️  Press Enter to return to menu...")
                
                else:
                    print("❌ Invalid choice. Please select 0-5.")
                    input("⏸️  Press Enter to continue...")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                input("⏸️  Press Enter to continue...")

    def run_command_line(self, args):
        """Run tests from command line arguments."""
        if len(args) < 2:
            print("Usage: python run_tests.py [test_name|all|summary]")
            print("Available tests: img, html, content, all, summary")
            return
        
        test_arg = args[1].lower()
        
        if test_arg == "img":
            self.load_and_run_test("test_img_agent")
        elif test_arg == "html":
            self.load_and_run_test("test_html_agent")
        elif test_arg == "content":
            self.load_and_run_test("test_content_agent")
        elif test_arg == "all":
            self.run_all_tests_sequential()
        elif test_arg == "summary":
            self.run_all_tests_with_summaries()
        else:
            print(f"❌ Unknown test: {test_arg}")
            print("Available tests: img, html, content, all, summary")


def main():
    """Main function to run the test runner."""
    runner = TestRunner()
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        runner.run_command_line(sys.argv)
    else:
        runner.run_interactive_menu()


if __name__ == "__main__":
    main() 