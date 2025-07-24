import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.tools.apply import ApplyTool
from agents.tools.enums import ApplyType
from database.content_chunk_db import ContentChunkDB
from langchain_google_genai import ChatGoogleGenerativeAI
from new_logger import get_logger

logger = get_logger()

class TestApplyTool(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        logger.info(f"\n===== Starting test: {self._testMethodName} =====")
        self.mock_model = MagicMock(spec=ChatGoogleGenerativeAI)
        self.mock_db = MagicMock(spec=ContentChunkDB)
        self.apply_tool = ApplyTool(content_db=self.mock_db, model="models/gemini-1.5-flash")
        self.apply_tool.apply_agent.model = self.mock_model

    def tearDown(self):
        logger.info(f"===== Finished test: {self._testMethodName} =====\n")

    def _log_test_pass(self):
        logger.info(f"Test PASSED: {self._testMethodName}")

    def _log_test_fail(self, exc):
        logger.error(f"Test FAILED: {self._testMethodName} - {exc}")

    def run(self, result=None):
        try:
            super().run(result)
            self._log_test_pass()
        except AssertionError as e:
            self._log_test_fail(e)
            raise
        except Exception as e:
            logger.error(f"Test ERROR: {self._testMethodName} - {e}")
            raise

    def _configure_mock_model(self, response_json: dict):
        """Helper function to configure the mock model's responses."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(response_json)
        self.mock_model.invoke.return_value = mock_response

    def _configure_mock_db(self, chunk_html: str):
        """Helper function to configure the mock database's responses."""
        mock_chunk = MagicMock()
        mock_chunk.html = chunk_html
        self.mock_db.load_content_chunk.return_value = mock_chunk

    def test_successful_insert(self):
        """Test a successful INSERT operation."""
        logger.info("Starting test_successful_insert")
        self._configure_mock_model({"position_id": "pos-123", "relative_position": "AFTER"})
        self._configure_mock_db("<p>New Content</p>")

        result = self.apply_tool.apply(
            type="INSERT",
            chunk_id="chunk-abc",
            document_structure="<div position_id='pos-123'></div>",
            last_prompt="add a new paragraph"
        )

        self.assertEqual(result, {"position_id": "pos-123", "relative_position": "AFTER"})

    def test_successful_delete(self):
        """Test a successful DELETE operation."""
        logger.info("Starting test_successful_delete")
        self._configure_mock_model({"position_id": "pos-456"})

        result = self.apply_tool.apply(
            type="DELETE",
            chunk_id="",
            document_structure="<div position_id='pos-456'></div>",
            last_prompt="remove the second paragraph"
        )

        self.assertEqual(result, {"position_id": "pos-456"})

    def test_successful_edit(self):
        """Test a successful EDIT operation."""
        logger.info("Starting test_successful_edit")
        self._configure_mock_model({"position_id": "pos-789"})
        self._configure_mock_db("<p>Updated Content</p>")

        result = self.apply_tool.apply(
            type="EDIT",
            chunk_id="chunk-def",
            document_structure="<div position_id='pos-789'></div>",
            last_prompt="change the third paragraph"
        )

        self.assertEqual(result, {"position_id": "pos-789"})

    def test_invalid_apply_type(self):
        """Test an invalid apply type."""
        logger.info("Starting test_invalid_apply_type")
        result = self.apply_tool.apply(
            type="INVALID",
            chunk_id="chunk-abc",
            document_structure="<div position_id='pos-123'></div>",
            last_prompt="do something invalid"
        )

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid apply type: INVALID")

    def test_chunk_not_found(self):
        """Test when a chunk is not found in the database."""
        logger.info("Starting test_chunk_not_found")
        self.mock_db.load_content_chunk.return_value = None

        result = self.apply_tool.apply(
            type="INSERT",
            chunk_id="chunk-nonexistent",
            document_structure="<div position_id='pos-123'></div>",
            last_prompt="add a new paragraph"
        )

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Chunk with id chunk-nonexistent not found.")

    def test_llm_error(self):
        """Test when the LLM returns an error."""
        logger.info("Starting test_llm_error")
        self.mock_model.invoke.side_effect = Exception("LLM Error")
        self._configure_mock_db("<p>New Content</p>")

        result = self.apply_tool.apply(
            type="INSERT",
            chunk_id="chunk-abc",
            document_structure="<div position_id='pos-123'></div>",
            last_prompt="add a new paragraph"
        )

        self.assertIn("error", result)
        self.assertIn("Failed to parse LLM response", result["error"])

    @unittest.skipIf(os.environ.get("RUN_INTEGRATION_TESTS") != "true", "Skipping integration test")
    def test_insert_with_real_llm(self):
        """Test a successful INSERT operation with a real LLM call."""
        logger.info("Starting test_insert_with_real_llm")
        # This test will make a real API call to the LLM
        apply_tool = ApplyTool(content_db=self.mock_db, model="models/gemini-1.5-flash")
        self._configure_mock_db("<p>New Content</p>")

        result = apply_tool.apply(
            type="INSERT",
            chunk_id="chunk-abc",
            document_structure="<div position_id='pos-123'></div><div position_id='pos-456'></div>",
            last_prompt="add a new paragraph after the first one"
        )

        self.assertIn("position_id", result)
        self.assertIn("relative_position", result)
        self.assertEqual(result["position_id"], "pos-123")
        self.assertEqual(result["relative_position"], "AFTER")


if __name__ == '__main__':
    unittest.main()
