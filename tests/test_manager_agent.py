import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.manager import ManagerAgent
from agents.tools.enums import ApplyType
from database.content_chunk_db import ContentChunk, ContentChunkDB
from context.store import ContextStore
from new_logger import get_logger

logger = get_logger()

class TestManagerAgentApplyTool(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        logger.info(f"\n===== Starting test: {self._testMethodName} =====")
        
        # Mock dependencies
        self.mock_model = MagicMock()
        self.mock_content_agent = MagicMock()
        self.mock_content_agent.chunk_db = MagicMock(spec=ContentChunkDB)
        self.mock_find_location_tool = MagicMock()
        self.mock_queue = MagicMock()

        # Instantiate ManagerAgent with mocked dependencies
        self.manager_agent = ManagerAgent(
            checkpoint_path=":memory:",
            model="dummy-model",
            store=ContextStore(),
            content_agent=self.mock_content_agent,
            queue=self.mock_queue
        )
        # Replace the real tool with our mock
        self.manager_agent.find_location_tool = self.mock_find_location_tool
        
        # Set common test data
        self.manager_agent.document_structure = "<doc>some content</doc>"
        self.manager_agent.last_prompt = "test prompt"

    def tearDown(self):
        """Tear down the test environment after each test."""
        logger.info(f"===== Finished test: {self._testMethodName} =====")

    @patch('agents.manager.interrupt')
    def test_apply_tool_func_successful_insert(self, mock_interrupt):
        """Test a successful INSERT operation in apply_tool_func."""
        # --- Setup ---
        action_type = "INSERT"
        chunk_id = "chunk-123"
        chunk_html = "<p>New Content</p>"
        
        # Mock the content chunk and its loading
        mock_chunk = ContentChunk(chunk_id=chunk_id, html=chunk_html)
        self.mock_content_agent.chunk_db.load_content_chunk.return_value = mock_chunk

        # Mock the result from the find_location_tool
        self.mock_find_location_tool.run.return_value = {
            "status": "success",
            "data_position_id_start": "pos-1",
            "data_position_id_end": "pos-1"
        }

        # --- Action ---
        result = self.manager_agent.apply_tool_func(action_type=action_type, chunk_id=chunk_id)

        # --- Verification ---
        # Verify that the find_location_tool was called correctly
        self.mock_find_location_tool.run.assert_called_once_with(
            apply_type=ApplyType.INSERT,
            document_structure=self.manager_agent.document_structure,
            last_prompt=self.manager_agent.last_prompt,
            chunk_id=chunk_id,
            chunk_html=chunk_html
        )

        # Verify that the interrupt was called with the correct payload
        mock_interrupt.assert_called_once()
        interrupt_args = mock_interrupt.call_args[0][0]
        self.assertEqual(interrupt_args["tool_name"], "apply")
        self.assertEqual(interrupt_args["action"], action_type)
        self.assertEqual(interrupt_args["data"]["chunk_html"], chunk_html)
        self.assertEqual(interrupt_args["data"]["data_position_id_start"], "pos-1")

        # Verify the status update queue was called
        self.mock_queue.put_nowait.assert_any_call(json.dumps({"type":"status_update","process":"FIND_LOCATION","status":"begin", "chunk_id": chunk_id,"apply_type":action_type}))
        self.mock_queue.put_nowait.assert_any_call(json.dumps({"type":"status_update","process":"FIND_LOCATION","status":"end", "chunk_id": chunk_id,"apply_type":action_type}))

        # Verify the function returns a success message
        self.assertEqual(result, {"status": "success", "message": "Apply request sent to client."})

    @patch('agents.manager.interrupt')
    def test_apply_tool_func_successful_delete(self, mock_interrupt):
        """Test a successful DELETE operation."""
        # --- Setup ---
        action_type = "DELETE"
        
        self.mock_find_location_tool.run.return_value = {
            "status": "success",
            "data_position_id_start": "pos-2",
            "data_position_id_end": "pos-3"
        }

        # --- Action ---
        result = self.manager_agent.apply_tool_func(action_type=action_type)

        # --- Verification ---
        self.mock_find_location_tool.run.assert_called_once_with(
            apply_type=ApplyType.DELETE,
            document_structure=self.manager_agent.document_structure,
            last_prompt=self.manager_agent.last_prompt,
            chunk_id="",
            chunk_html=""
        )

        mock_interrupt.assert_called_once()
        interrupt_args = mock_interrupt.call_args[0][0]
        self.assertEqual(interrupt_args["action"], "DELETE")
        self.assertEqual(interrupt_args["data"]["chunk_html"], "") # No chunk_html for delete
        self.assertEqual(interrupt_args["data"]["data_position_id_start"], "pos-2")

        self.assertEqual(result, {"status": "success", "message": "Apply request sent to client."})

    def test_apply_tool_func_invalid_type(self):
        """Test apply_tool_func with an invalid action type."""
        result = self.manager_agent.apply_tool_func(action_type="INVALID_TYPE")
        self.assertEqual(result, {"status": "error", "message": "Invalid apply type: INVALID_TYPE"})

    def test_apply_tool_func_chunk_not_found(self):
        """Test apply_tool_func when the chunk_id is required but not found."""
        self.mock_content_agent.chunk_db.load_content_chunk.return_value = None
        
        result = self.manager_agent.apply_tool_func(action_type="INSERT", chunk_id="non-existent-chunk")
        
        self.assertEqual(result, {"status": "error", "message": "Chunk with id non-existent-chunk not found."})

    def test_apply_tool_func_find_location_fails(self):
        """Test when the find_location_tool returns an error."""
        # --- Setup ---
        action_type = "DELETE"
        self.mock_find_location_tool.run.return_value = {"status": "error"}

        # --- Action ---
        result = self.manager_agent.apply_tool_func(action_type=action_type)

        # --- Verification ---
        self.assertEqual(result, {"status": "error", "message": "Failed to find location for the requested action."})

if __name__ == '__main__':
    unittest.main(verbosity=2)