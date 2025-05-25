# Content Agent System - Test Suite

This directory contains comprehensive test suites for all components of the Content Agent System.

## 🧪 Test Structure

### Test Files

- **`test_img_agent.py`** - Image Agent functionality tests
  - Image processing and manipulation
  - AI-powered image analysis and captioning
  - Image management and storage operations
  - Error handling and edge cases

- **`test_html_agent.py`** - HTML Agent functionality tests
  - HTML content generation and validation
  - Document integration and context management
  - Style compliance and formatting
  - Quality evaluation and retry logic
  - Performance and complexity testing
  - Context generation improvements

- **`test_content_agent.py`** - Content Agent coordination tests
  - Multi-agent coordination and workflow management
  - Content strategy and planning
  - Integration between HTML and Image agents
  - Quality assurance and complex workflows

- **`run_tests.py`** - Test runner with interactive menu
  - Run individual test suites
  - Execute all tests sequentially
  - Comprehensive reporting and summaries

## 🚀 Quick Start

### Interactive Test Runner (Recommended)
```bash
cd tests
python run_tests.py
```

This launches an interactive menu where you can:
- Select specific test suites to run
- Run all tests with detailed summaries
- Get comprehensive performance reports

### Command Line Usage
```bash
# Run specific test suites
python tests/run_tests.py img      # Image Agent tests
python tests/run_tests.py html     # HTML Agent tests
python tests/run_tests.py content  # Content Agent tests
python tests/run_tests.py all      # All tests sequential
python tests/run_tests.py summary  # All tests with summaries

# Run individual test files directly
python tests/test_img_agent.py
python tests/test_html_agent.py
python tests/test_content_agent.py
```

## 📋 Test Categories

### Image Agent Tests
1. **Basic Operations** - Image storage, retrieval, search
2. **AI Analysis** - Caption generation, content analysis, OCR
3. **Image Processing** - Resize, filters, enhancement, format conversion
4. **Advanced Operations** - Batch processing, comparison, validation
5. **Error Handling** - Invalid inputs, edge cases
6. **Interactive Mode** - Manual testing interface

### HTML Agent Tests
1. **Basic Generation** - Simple HTML content creation
2. **Complex Generation** - Multi-section content with tables and lists
3. **Validation & Retry** - Quality control and improvement loops
4. **Style Compliance** - Adherence to formatting requirements
5. **Edge Cases** - Special characters, unicode, empty inputs
6. **Performance Metrics** - Speed and quality measurements

### Content Agent Tests
1. **HTML Coordination** - Integration with HTML agent
2. **Image Coordination** - Integration with image agent
3. **Context Analysis** - Understanding available resources
4. **Coordinated Generation** - Multi-agent workflows
5. **Strategy Planning** - Content planning and requirements analysis
6. **Quality Assurance** - End-to-end quality control
7. **Complex Workflows** - Multi-step content creation processes

## 🎯 Test Features

### Interactive Testing
- Step-by-step guided testing
- User input for specific test cases
- Real-time feedback and results
- Pause between tests for review

### Comprehensive Coverage
- Success scenarios and error handling
- Performance measurements
- Quality validation
- Edge case testing

### Detailed Reporting
- Test execution summaries
- Performance metrics
- Success/failure rates
- Detailed error information

## 📊 Understanding Test Results

### Success Indicators
- ✅ **PASS** - Test completed successfully
- 🎯 **Score** - Quality score (for HTML agent)
- ⏱️ **Time** - Execution time
- 🔄 **Retries** - Number of retry attempts

### Error Indicators
- ❌ **FAIL** - Test failed
- ⚠️ **Warning** - Non-critical issues
- 💡 **Recommendations** - Suggested fixes

## 🔧 Prerequisites

### Environment Setup
1. **API Keys** - Google API key in `.env` file
2. **Dependencies** - All Python packages installed
3. **Database Access** - Write permissions for SQLite files

### Required Environment Variables
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## 💡 Tips for Testing

### First Time Setup
1. Start with Image Agent tests to verify basic functionality
2. Ensure you have a stable internet connection for image URLs
3. Have some test image URLs ready for manual testing

### Troubleshooting
- **API Errors**: Check your Google API key and quota
- **Database Errors**: Ensure write permissions in the project directory
- **Import Errors**: Verify all dependencies are installed
- **Network Errors**: Check internet connection for image downloads

### Best Practices
- Run tests in a clean environment
- Review test output carefully
- Use interactive mode for detailed exploration
- Check the comprehensive summaries for overall system health

## 📚 Additional Resources

- **System Overview**: See `../IMPROVEMENTS_SUMMARY.md`
- **Agent Documentation**: Check individual agent files
- **Error Logs**: Review console output for detailed error information

## 🎉 Expected Results

When all tests pass, you should see:
- ✅ All test suites completing successfully
- 🎯 High quality scores for HTML generation
- ⚡ Reasonable performance times
- 🔄 Minimal retry attempts needed
- 📊 High overall success rates

This indicates your Content Agent System is working correctly and ready for production use! 