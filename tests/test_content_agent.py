import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.content import ContentAgent
from database.content_chunk_db import ContentChunkDB, ContentChunk

class TestContentAgent(unittest.TestCase):

    def setUp(self):
        """Set up the test environment for each test."""
        # This creates a new in-memory SQLite database for each test
        self.content_agent = ContentAgent(
            model="models/gemini-pro",  # This will be mocked in most tests
            checkpoint_path=":memory:"
        )
        # Mock the underlying generative model for the agent
        self.mock_model = MagicMock()
        self.content_agent.model_instance = self.mock_model

        # We also need to mock the agents that the ContentAgent uses
        self.mock_html_agent = MagicMock()
        self.content_agent.html_agent = self.mock_html_agent

        self.mock_image_agent = MagicMock()
        self.content_agent.image_agent = self.mock_image_agent

    def test_initialization(self):
        """Test that the ContentAgent initializes correctly."""
        self.assertIsInstance(self.content_agent, ContentAgent)
        self.assertIsInstance(self.content_agent.chunk_db, ContentChunkDB)
        self.assertIsNotNone(self.content_agent.agent)

    def test_run_html_agent_tool_success(self):
        """Test the successful execution of the `run_html_agent` tool."""
        # Configure the mock HTML agent to return a successful result
        test_html = "<p><span>This is successful HTML.</span></p>"
        self.mock_html_agent.run.return_value = {
            "status": "success",
            "html": test_html
        }

        # Call the tool method directly
        result_str = self.content_agent.run_html_agent("A test", "default style")

        # Assertions on the string returned by the tool
        self.assertIn("HTML Generation Result", result_str)
        self.assertIn("chunk_id", result_str)
        self.assertIn("Status: PENDING", result_str)

        # Assertions on the internal state
        self.assertEqual(len(self.content_agent.generated_chunks), 1)
        chunk = self.content_agent.generated_chunks[0]
        self.assertEqual(chunk['html'], test_html)
        self.assertEqual(chunk['status'], 'PENDING')

        # Verify that the chunk was saved to the DB
        saved_chunk = self.content_agent.chunk_db.get_chunk_by_id(chunk['id'])
        self.assertIsNotNone(saved_chunk)
        self.assertEqual(saved_chunk.html, test_html)

    def test_run_html_agent_tool_failure(self):
        """Test the `run_html_agent` tool when the html_agent reports an error."""
        # Configure the mock HTML agent to return an error status
        error_html = "<p><span>Generation failed.</span></p>"
        self.mock_html_agent.run.return_value = {
            "status": "error",
            "html": error_html
        }

        # Call the tool method
        result_str = self.content_agent.run_html_agent("A failing test", "error style")

        # Assertions on the string returned by the tool
        self.assertIn("HTML Generation failed", result_str)
        self.assertIn("Created a placeholder chunk", result_str)

        # Assertions on the internal state
        self.assertEqual(len(self.content_agent.generated_chunks), 1)
        chunk = self.content_agent.generated_chunks[0]
        self.assertEqual(chunk['html'], "<p><span>Error generating content.</span></p>")
        self.assertEqual(chunk['status'], 'ERROR')

        # Verify the error chunk was saved to the DB
        saved_chunk = self.content_agent.chunk_db.get_chunk_by_id(chunk['id'])
        self.assertIsNotNone(saved_chunk)
        self.assertEqual(saved_chunk.status, 'ERROR')

    def test_run_html_agent_tool_exception(self):
        """Test the `run_html_agent` tool when the html_agent raises an exception."""
        # Configure the mock HTML agent to raise an exception
        self.mock_html_agent.run.side_effect = Exception("A critical failure occurred")

        # Call the tool method
        result_str = self.content_agent.run_html_agent("An exception test", "exception style")

        # Assertions on the string returned by the tool
        self.assertIn("Error in HTML generation", result_str)
        self.assertIn("A critical failure occurred", result_str)

        # Assertions on the internal state (it should still create an error chunk)
        self.assertEqual(len(self.content_agent.generated_chunks), 1)
        chunk = self.content_agent.generated_chunks[0]
        self.assertIn("An exception occurred", chunk['html'])
        self.assertEqual(chunk['status'], 'ERROR')

    @patch('langgraph.prebuilt.create_react_agent')
    def test_agent_run_returns_no_chunks(self, mock_create_agent):
        """Test the main `run` method when the agent generates no chunks."""
        # Mock the entire react agent to simulate it not calling any tools
        mock_react_agent = MagicMock()
        mock_react_agent.invoke.return_value = {"messages": [{"role": "assistant", "content": "I did nothing."}]}
        mock_create_agent.return_value = mock_react_agent

        # Re-initialize the agent to use the mocked react agent
        self.content_agent.agent = mock_react_agent

        # Run the agent
        result = self.content_agent.run("A prompt that does nothing")

        # Assertions
        self.assertEqual(result, "No content chunks were generated. Please check the logs for errors.")
        self.assertEqual(len(self.content_agent.generated_chunks), 0)

    @patch('langgraph.prebuilt.create_react_agent')
    def test_agent_run_returns_only_error_chunks(self, mock_create_agent):
        """Test the `run` method when all generated chunks are errors."""
        # This test will actually call our tool, but the tool will report an error
        self.mock_html_agent.run.return_value = {"status": "error", "html": ""}

        # We need to mock the agent's response to simulate it calling our tool
        mock_react_agent = MagicMock()
        # The tool's output will be added to the state, so we check the final state
        # Here we just need to ensure the logic inside `run` that checks the chunks works
        mock_react_agent.invoke.return_value = "some response" # The response itself doesn't matter
        self.content_agent.agent = mock_react_agent

        # Manually create an error chunk as if the tool was called
        error_chunk = ContentChunk(html="error", status="ERROR")
        self.content_agent.generated_chunks.append(error_chunk.to_dict())

        # Run the agent
        result = self.content_agent.run("A prompt that fails")

        # Assertion
        self.assertEqual(result, "All content generation attempts resulted in errors. Please review your request and the agent's capabilities.")

    def test_multiple_html_agent_calls_mixed_results(self):
        """Test a scenario with multiple calls to the html_agent tool with mixed results."""
        # Configure the mock to return different results on subsequent calls
        self.mock_html_agent.run.side_effect = [
            {"status": "success", "html": "<p><span>First call success.</span></p>"},
            {"status": "error", "html": ""},
            {"status": "success", "html": "<div><span>Third call success.</span></div>"}
        ]

        # Simulate three calls to the tool
        res1 = self.content_agent.run_html_agent("First", "style")
        res2 = self.content_agent.run_html_agent("Second", "style")
        res3 = self.content_agent.run_html_agent("Third", "style")

        # Assertions on the internal state
        self.assertEqual(len(self.content_agent.generated_chunks), 3)

        # Check chunk 1 (success)
        self.assertEqual(self.content_agent.generated_chunks[0]['status'], 'PENDING')
        self.assertIn("First call success", self.content_agent.generated_chunks[0]['html'])

        # Check chunk 2 (error)
        self.assertEqual(self.content_agent.generated_chunks[1]['status'], 'ERROR')
        self.assertIn("Error generating content", self.content_agent.generated_chunks[1]['html'])

        # Check chunk 3 (success)
        self.assertEqual(self.content_agent.generated_chunks[2]['status'], 'PENDING')
        self.assertIn("Third call success", self.content_agent.generated_chunks[2]['html'])

        # Verify the DB state
        all_chunks = self.content_agent.chunk_db.get_all_chunks()
        self.assertEqual(len(all_chunks), 3)
        
        # Sort by creation time to ensure order
        all_chunks.sort(key=lambda c: c.created_at)
        
        self.assertEqual(all_chunks[0].status, 'PENDING')
        self.assertEqual(all_chunks[1].status, 'ERROR')
        self.assertEqual(all_chunks[2].status, 'PENDING')

if __name__ == '__main__':
    unittest.main()
