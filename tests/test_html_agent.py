import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.sub_agents.html import HtmlAgent
from langchain_google_genai import ChatGoogleGenerativeAI

class TestHtmlAgent(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        # Use a mock for the generative model to avoid actual API calls
        self.mock_model = MagicMock(spec=ChatGoogleGenerativeAI)
        self.html_agent = HtmlAgent(
            model=self.mock_model,
            checkpoint_path=":memory:",  # Use in-memory database for checkpoints
            acceptance_threshold=85,
            max_retries=2
        )

    def test_initialization(self):
        """Test that the HtmlAgent initializes correctly."""
        self.assertIsInstance(self.html_agent, HtmlAgent)
        self.assertEqual(self.html_agent.acceptance_threshold, 85)
        self.assertEqual(self.html_agent.max_retries, 2)
        self.assertIsNotNone(self.html_agent.graph)

    def _configure_mock_model(self, generated_html, score, feedback):
        """Helper function to configure the mock model's responses."""
        # Mock the response for the content generation call
        self.mock_model.invoke.return_value = MagicMock(content=generated_html)

        # Mock the response for the structured output (evaluation) call
        self.mock_model.with_structured_output.return_value.invoke.return_value = MagicMock(
            score=score,
            feedback=feedback
        )

    def test_successful_html_generation(self):
        """Test a successful HTML generation workflow."""
        # Configure the mock model for a successful run
        test_html = "<p><span>This is a test paragraph.</span></p>"
        self._configure_mock_model(
            generated_html=test_html,
            score=95,  # Score above the acceptance threshold
            feedback="Excellent content, meets all criteria."
        )

        # Define inputs for the agent
        description = "Create a simple test paragraph."
        style_guidelines = "Use standard paragraph and span tags."

        # Run the agent
        result = self.html_agent.run(description, style_guidelines)

        # Assertions to verify the outcome
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'success')
        self.assertIn('html', result)
        # The validator might slightly modify the HTML, so check for key parts
        self.assertIn("This is a test paragraph", result['html'])
        self.assertIn("<p>", result['html'])
        self.assertIn("<span>", result['html'])

    def test_empty_html_from_model(self):
        """Test how the agent handles an empty HTML string from the model."""
        # Configure the mock model to return empty content
        self._configure_mock_model(
            generated_html="",
            score=0,  # Evaluation would likely fail
            feedback="No content generated."
        )

        description = "An empty paragraph."
        style_guidelines = "Standard styles."

        # Run the agent
        result = self.html_agent.run(description, style_guidelines)

        # The agent should not return a success status
        self.assertNotEqual(result['status'], 'success')
        # Even on failure, it should return some valid HTML structure
        self.assertTrue(len(result['html']) > 0)
        self.assertIn("No satisfactory HTML generated", result['html'])

    def test_generation_with_retries(self):
        """Test the retry mechanism when the initial score is too low."""
        # First attempt: low score
        low_score_html = "<p><span>Needs improvement.</span></p>"
        # Second attempt: high score
        high_score_html = "<p><span>This is much better.</span></p>"

        # Configure the model to return low score first, then high score
        self.mock_model.invoke.side_effect = [
            MagicMock(content=low_score_html),
            MagicMock(content=high_score_html)
        ]
        self.mock_model.with_structured_output.return_value.invoke.side_effect = [
            MagicMock(score=70, feedback="Content is too simple."),
            MagicMock(score=90, feedback="Great improvement!")
        ]

        description = "A paragraph that requires a retry."
        style_guidelines = "Initial attempt should be simple, second better."

        # Run the agent
        result = self.html_agent.run(description, style_guidelines)

        # Assertions
        self.assertEqual(result['status'], 'success')
        self.assertIn("This is much better", result['html'])
        # Check that the model was called twice for generation
        self.assertEqual(self.mock_model.invoke.call_count, 2)

    def test_max_retries_reached(self):
        """Test what happens when the agent never meets the acceptance threshold."""
        # Configure the model to consistently return a low score
        self._configure_mock_model(
            generated_html="<p><span>Always failing.</span></p>",
            score=50,
            feedback="Does not meet requirements."
        )

        description = "Content that will never pass evaluation."
        style_guidelines = "Strict requirements."

        # Run the agent
        result = self.html_agent.run(description, style_guidelines)

        # Assertions
        self.assertNotEqual(result['status'], 'success')
        # It should return the best HTML it managed to generate
        self.assertIn("Always failing", result['html'])
        # The number of generation attempts should be max_retries + 1
        self.assertEqual(self.mock_model.invoke.call_count, self.html_agent.max_retries + 1)

    def test_validation_failure(self):
        """Test the agent's response to invalid HTML from the model."""
        # Configure the model to return malformed HTML
        malformed_html = "<p><span>This is not properly closed."
        self._configure_mock_model(
            generated_html=malformed_html,
            score=90,  # This score won't be used if validation fails first
            feedback="N/A"
        )

        description = "Generate malformed HTML."
        style_guidelines = "No specific styles."

        # Run the agent
        result = self.html_agent.run(description, style_guidelines)

        # Assertions
        self.assertNotEqual(result['status'], 'success')
        # The validator should attempt to fix the HTML, but the flow should still reject it
        # The final output should be the "no satisfactory HTML" message
        self.assertIn("No satisfactory HTML generated", result['html'])

    def test_model_returns_none(self):
        """Test how the agent handles a None response from the model."""
        self._configure_mock_model(
            generated_html=None,
            score=0,
            feedback="No content generated."
        )

        description = "A paragraph that results in None."
        style_guidelines = "Standard styles."

        result = self.html_agent.run(description, style_guidelines)

        self.assertNotEqual(result['status'], 'success')
        self.assertTrue(len(result['html']) > 0)
        self.assertIn("No satisfactory HTML generated", result['html'])

    def test_model_returns_whitespace(self):
        """Test how the agent handles a whitespace-only response from the model."""
        self._configure_mock_model(
            generated_html="   \t\n  ",
            score=0,
            feedback="No content generated."
        )

        description = "A paragraph that results in only whitespace."
        style_guidelines = "Standard styles."

        result = self.html_agent.run(description, style_guidelines)

        self.assertNotEqual(result['status'], 'success')
        self.assertTrue(len(result['html']) > 0)
        self.assertIn("No satisfactory HTML generated", result['html'])

    def test_model_returns_empty_html_structure(self):
        """Test how the agent handles an empty HTML structure from the model."""
        self._configure_mock_model(
            generated_html="<p><span></span></p>",
            score=10, # low score for empty content
            feedback="Content is empty."
        )

        description = "An empty paragraph."
        style_guidelines = "Standard styles."

        result = self.html_agent.run(description, style_guidelines)

        self.assertNotEqual(result['status'], 'success')
        self.assertTrue(len(result['html']) > 0)
        self.assertIn("No satisfactory HTML generated", result['html'])

if __name__ == '__main__':
    unittest.main()
