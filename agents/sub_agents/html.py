from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.errors import GraphRecursionError  # Import the exception
from pydantic import BaseModel
from utils.html_validator import HTMLValidator
import os
from typing import Annotated, List, Optional
from IPython.display import Image, display
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from context.store import ContextStore
from time import sleep
from new_logger import get_logger

logger = get_logger()

# Define State with cleaner structure
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    html: str
    description: str
    style_guidelines: str
    context: str  # Single consolidated context string
    
    document_structure: str

    content_generation_outcome: str
    validation_outcome: str
    evaluator_score: int
    evaluator_feedback: str
    
    # Retry management
    current_retry_count: int
    best_html_so_far: str
    best_score_so_far: int
    max_retries_reached: bool

class EvaluatorResponse(BaseModel):
    score: int
    feedback: str

class HtmlAgent:
    def __init__(self,
                 model: ChatGoogleGenerativeAI,
                 checkpoint_path: str = "data/checkpoint.sqlite",
                 disallowed_tags=["script", "iframe", "style", "link", "meta", "head", "body", "html", "div", "em", "br", "i", "b"],
                 allowed_tags=["p", "span", "u", "ol", "ul", "li", "table", "tr", "td", "th", "tbody", "h1", "h2", "h3", "h4", "h5", "h6"],
                 acceptance_threshold=90,
                 max_retries=3,
                 debug=True):
        
        self.model = model
        self.html_validator = HTMLValidator()
        self.connection = MemorySaver()
        self.debug = debug
        
        self.disallowed_tags = disallowed_tags
        self.allowed_tags = allowed_tags
        self.acceptance_threshold = acceptance_threshold
        self.max_retries = max_retries
        
        # Build the graph
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("moderator", self.moderator_action)
        graph_builder.add_node("content_generator", self.content_generator_action)
        graph_builder.add_node("html_validator_node", self.html_validator_action)
        graph_builder.add_node("evaluator", self.evaluator_action)
        graph_builder.add_node("handle_content_error", self.handle_content_error_action)
        graph_builder.add_node("prepare_final_output", self.prepare_final_output_action)
        
        # Define edges
        graph_builder.add_edge(START, "moderator")
        
        # Conditional edges
        graph_builder.add_conditional_edges(
            "moderator",
            self.decide_moderator_next_step,
            {
                "generate_content": "content_generator",
                "end_with_best": "prepare_final_output"
            }
        )
        
        graph_builder.add_conditional_edges(
            "content_generator",
            self.check_content_generation_outcome,
            {
                "success": "html_validator_node",
                "error": "handle_content_error"
            }
        )
        
        graph_builder.add_edge("handle_content_error", "moderator")
        
        graph_builder.add_conditional_edges(
            "html_validator_node",
            self.check_validation_outcome,
            {
                "success": "evaluator",
                "error": "moderator"
            }
        )
        
        graph_builder.add_conditional_edges(
            "evaluator",
            self.check_evaluation_score,
            {
                "accept": END,
                "reject": "moderator"
            }
        )
        
        graph_builder.add_edge("prepare_final_output", END)
        
        self.graph = graph_builder.compile(checkpointer=self.connection)
        # Set a default config with a thread_id. We will add the recursion limit at runtime.
        self.base_config = {"configurable": {"thread_id": "1"}}

    def get_context(self, description: str, style_guidelines: str = "", previous_context: str = "") -> str:
        """
        Create a comprehensive context string that includes:
        - Task description and style guidelines
        - Available documents from DocManager
        - Previous context/history
        - Available images summary
        """
        context_parts = []
        
        # Add task information
        context_parts.append("=== TASK INFORMATION ===")
        context_parts.append(f"Description: {description}")
        context_parts.append(f"Style Guidelines: {style_guidelines}")
        context_parts.append("")
        
        # Add previous context/history
        if previous_context:
            context_parts.append("=== PREVIOUS CONTEXT ===")
            context_parts.append(previous_context)
            context_parts.append("")
        
        return "\n".join(context_parts)

    def get_content_generation_prompt(self, state: State) -> str:
        """
        Generate a comprehensive prompt for HTML content generation.
        """
        context = state.get("context", "")
        description = state.get("description", "")
        style_guidelines = state.get("style_guidelines", "")
        document_structure = state.get("document_structure", "")
        
        prompt = f"""You are an expert HTML content generator. Your task is to create high-quality HTML content based on the provided information.

=== CONTENT GENERATION RULES ===
{self._get_content_generation_rules()}

=== CONTEXT INFORMATION ===
{context}

=== DOCUMENT STRUCTURE ===
{document_structure}
=== GENERATION TASK ===
Create HTML content that fulfills the following requirements:
- Description: {description}
- Style Guidelines: {style_guidelines}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY the HTML content - no explanations, markdown formatting, or code blocks
2. Start directly with the first HTML tag (e.g., <p>, <h1>, etc.)
3. Generate NEW, original content based on the description and guidelines
4. Use the context information to inform your content but create fresh material
5. Ensure all text is properly wrapped in <span> tags within block elements
6. Apply styles using inline style attributes only

Generate the HTML content now:"""
        
        return prompt

    def _get_content_generation_rules(self) -> str:
        """Define the HTML generation rules."""
        return f"""
1. Valid Structure: Ensure perfect HTML syntax with properly nested tags
2. Allowed Tags ONLY: Use ONLY these tags: {', '.join(self.allowed_tags)}
3. Forbidden Tags: NEVER use these tags: {', '.join(self.disallowed_tags)}
4. Text Encapsulation: ALL visible text MUST be wrapped in <span> tags
5. Span Placement: Place <span> tags immediately inside block elements (p, h1-h6, li, td, th)
6. Root Element: Start with appropriate block element (p, h1-h6, table, ul, ol)
7. Tables: Must contain <tbody>, use <th> for headers, <td> for data cells
8. Colors: Use HEX format only (#FF0000, #333333, etc.)
9. Styling: Use inline style attributes only - no external CSS
10. Headings: Use h1-h6 appropriately for content hierarchy
11. Spacing: Use <p> tags for paragraphs, margins/padding for spacing
12. Font Styling: Apply general styles to block elements, use <span> for inline changes
13. Attributes: Ensure valid CSS properties in double quotes
14. Inline styles: Ensure the styles are inline. Inline styles in our case,in fact, are encouraged.
"""

    def run(self, description: str, style_guidelines: str, context: str = "", document_structure : str = ""):
        """
        Main entry point for HTML generation.
        """
        full_context = self.get_context(description, style_guidelines, context)
        initial_state = {
            "description": description,
            "style_guidelines": style_guidelines,
            "document_structure" : document_structure,
            "context": full_context,
            "html": "",
            "current_retry_count": 0,
            "best_html_so_far": "",
            "best_score_so_far": -1,
            "max_retries_reached": False,
            "messages": []
        }
        
        # Define the configuration for this run, including the recursion limit
        config = {**self.base_config, "recursion_limit": 15}
        
        custom_response = {}

        try:
            response = self.graph.invoke(initial_state, config)
            if self.debug:
                logger.debug(f"HTML Agent Response: {response}")
                logger.debug(f"Generated HTML: {response.get('html', '')}")
                logger.debug(f"Validation Outcome: {response.get('validation_outcome', '')}")
                logger.debug(f"Evaluator Feedback: {response.get('evaluator_feedback', '')}")
            
            error_html = "<p><span>Error: No satisfactory HTML generated</span></p>"
            custom_response = {
                "status": "success" if response.get("html") and response.get("validation_outcome") != "error" else "error",
                "html": response.get("html", "") if response.get("validation_outcome") != "error" else error_html
            }

        except GraphRecursionError:
            logger.error(f"Graph reached recursion limit of {config['recursion_limit']} without finishing.")
            custom_response = {
                "status": "error",
                "html": "<p><span>Error: No satisfactory HTML generated</span></p>"
            }
        
        except Exception as e:
            logger.error(f"An unexpected error occurred during graph execution: {e}")
            custom_response = {
                "status": "error",
                "html": f"<p><span>An unexpected error occurred: {e}</span></p>"
            }

        return custom_response

    def moderator_action(self, state: State) -> dict:
        """Moderator manages the flow and retry logic."""
        logger.info("--- Moderator Action ---")
        description = state.get("description", "")
        style_guidelines = state.get("style_guidelines", "")
        html = state.get("html","")
        logger.info(f"Description: {description}")
        logger.info(f"Style Guidelines: {style_guidelines}")
        
        current_retries = state.get("current_retry_count", 0)
        evaluator_feedback = state.get("evaluator_feedback")
        evaluator_score = state.get("evaluator_score")
        validation_outcome = state.get("validation_outcome")
        
        is_retry = (evaluator_score is None) or (validation_outcome == "error")
        
        if is_retry:
            current_retries += 1
        
        if current_retries > self.max_retries:
            logger.warning(f"Moderator: Max retries ({self.max_retries}) exceeded.")
            error_html = "<p><span>Error: LLM returned empty content</span></p>"
            return {
                "max_retries_reached": True,
                "current_retry_count": current_retries,
                "html": html if html != "" else error_html
            }
        
        updates = {
            "html": "",
            "evaluator_score": None,
            "evaluator_feedback": None,
            "content_generation_outcome": None,
            "validation_outcome": None,
            "current_retry_count": current_retries,
            "max_retries_reached": False
        }
        
        if evaluator_feedback and evaluator_score is not None and evaluator_score < self.acceptance_threshold:
            logger.info(f"Moderator: Retry {current_retries}/{self.max_retries} due to low score ({evaluator_score})")
            logger.info(f"Feedback: {evaluator_feedback}")
        elif validation_outcome == "error":
            logger.info(f"Moderator: Retry {current_retries}/{self.max_retries} due to validation error")
        else:
            logger.info(f"Moderator: Initial attempt ({current_retries}/{self.max_retries})")
        
        return updates

    def content_generator_action(self, state: State) -> dict:
        """Generate HTML content using the LLM."""
        logger.info("--- Content Generator Action ---")
        sleep(3)
        try:
            prompt = self.get_content_generation_prompt(state)
            response = self.model.invoke(prompt)
            generated_html = response.content.strip()
            
            if generated_html.startswith("```html"):
                generated_html = generated_html.replace("```html", "").replace("```", "").strip()
            elif generated_html.startswith("```"):
                generated_html = generated_html.replace("```", "").strip()
            
            logger.info(f"Generated HTML length: {len(generated_html)} characters")
            
            return {
                "html": generated_html,
                "content_generation_outcome": "success",
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": f"Generated HTML content ({len(generated_html)} chars)"}]
                )
            }
            
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return {
                "html": f"<p><span>Error during content generation: {str(e)}</span></p>",
                "content_generation_outcome": "error",
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": f"Error in content generation: {str(e)}"}]
                )
            }

    def html_validator_action(self, state: State) -> dict:
        """Validate the generated HTML."""
        logger.info("--- HTML Validator Action ---")
        sleep(3)
        html_to_validate = state.get("html", "")
        if not html_to_validate:
            return {
                "validation_outcome": "error",
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": "Validation error: No HTML content to validate"}]
                )
            }
        
        try:
            cleaned_html = self.html_validator._clean_llm_html_output(html_to_validate)
            validation_result = self.html_validator.validate_and_repair(cleaned_html)
            
            if validation_result["status"] == "error":
                logger.warning(f"Validation failed: {validation_result.get('message', 'Unknown error')}")
                return {
                    "validation_outcome": "error",
                    "html": validation_result.get("html", html_to_validate),
                    "messages": add_messages(
                        state.get("messages", []), 
                        [{"role": "assistant", "content": f"Validation Error: {validation_result.get('message', 'Unknown error')}"}]
                    )
                }
            
            logger.info("Validation successful")
            return {
                "html": validation_result["html"],
                "validation_outcome": "success",
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": "HTML validation successful"}]
                )
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "validation_outcome": "error",
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": f"Validation exception: {str(e)}"}]
                )
            }

    def evaluator_action(self, state: State) -> dict:
        """Evaluate the quality of generated HTML."""
        logger.info("--- Evaluator Action ---")
        sleep(2)
        try:
            prompt = self._get_evaluator_prompt(state)
            response = self.model.with_structured_output(EvaluatorResponse).invoke(prompt)
            
            current_html = state.get("html", "")
            best_html_so_far = state.get("best_html_so_far", "")
            best_score_so_far = state.get("best_score_so_far", -1)
            
            new_best_html = best_html_so_far
            new_best_score = best_score_so_far
            
            if response.score > best_score_so_far:
                new_best_score = response.score
                new_best_html = current_html
                logger.info(f"New best score: {new_best_score} (previous: {best_score_so_far})")
            
            logger.info(f"Evaluation score: {response.score}/{100}")
            logger.info(f"Feedback: {response.feedback}")
            
            return {
                "evaluator_score": response.score,
                "evaluator_feedback": response.feedback,
                "best_html_so_far": new_best_html,
                "best_score_so_far": new_best_score,
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": f"Evaluation: {response.score}/100 - {response.feedback}"}]
                )
            }
            
        except Exception as e:
            logger.error(f"Evaluator error: {e}")
            return {
                "evaluator_score": 0,
                "evaluator_feedback": f"Error during evaluation: {str(e)}",
                "best_html_so_far": state.get("best_html_so_far", ""),
                "best_score_so_far": state.get("best_score_so_far", -1),
                "messages": add_messages(
                    state.get("messages", []), 
                    [{"role": "assistant", "content": f"Evaluation error: {str(e)}"}]
                )
            }

    def _get_evaluator_prompt(self, state: State) -> str:
        """Generate evaluation prompt."""
        description = state.get("description", "")
        style_guidelines = state.get("style_guidelines", "")
        html_content = state.get("html", "")
        generation_rules = self._get_content_generation_rules()
        return f"""You are an expert HTML content evaluator. Evaluate the provided HTML content based on the task requirements and quality standards.
        The HTML content is generated as a part of a document editor such as google docs or microsoft word. Therefore, it shouldn't be treated as if it were a part of a website.

TASK REQUIREMENTS:
- Description: {description}
- Style Guidelines: {style_guidelines}

HTML CONTENT TO EVALUATE:
{html_content}

RULES GIVEN TO THE GENERATOR:
{generation_rules}

EVALUATION CRITERIA:
1. Task Fulfillment (40 points): Does the content match the description and requirements?
2. Style Compliance (30 points): Does it follow the style guidelines?
3. HTML Quality (20 points): Is the HTML well-formed, semantic, and properly structured?
4. Content Quality (10 points): Is the content engaging, clear, and well-written?

Provide a score from 0-100 and detailed feedback. If the score is below 100, explain specific areas for improvement.

Respond with a JSON object:
{{
    "score": <integer 0-100>,
    "feedback": "<detailed feedback string>"
}}"""

    def handle_content_error_action(self, state: State) -> dict:
        """Handle content generation errors."""
        logger.warning("--- Handle Content Error Action ---")
        return {
            "html": "<p><span>Content processing encountered an error.</span></p>",
            "messages": add_messages(
                state.get("messages", []), 
                [{"role": "assistant", "content": "Error: Content generation failed, retrying..."}]
            )
        }

    def prepare_final_output_action(self, state: State) -> dict:
        """Prepare final output when max retries reached."""
        logger.info("--- Prepare Final Output Action ---")
        best_html = state.get("best_html_so_far", "<p><span>No satisfactory HTML generated.</span></p>")
        best_score = state.get("best_score_so_far", 0)
        
        return {
            "html": best_html,
            "evaluator_score": best_score,
            "messages": add_messages(
                state.get("messages", []), 
                [{"role": "assistant", "content": f"Max retries reached. Using best HTML (score: {best_score})"}]
            )
        }

    # Condition functions
    def decide_moderator_next_step(self, state: State) -> str:
        if state.get("max_retries_reached", False):
            return "end_with_best"
        return "generate_content"

    def check_content_generation_outcome(self, state: State) -> str:
        outcome = state.get("content_generation_outcome", "error")
        logger.info(f"Content generation outcome: {outcome}")
        return outcome

    def check_validation_outcome(self, state: State) -> str:
        outcome = state.get("validation_outcome", "error")
        logger.info(f"Validation outcome: {outcome}")
        return outcome

    def check_evaluation_score(self, state: State) -> str:
        score = state.get("evaluator_score", 0)
        threshold = self.acceptance_threshold
        logger.info(f"Evaluation score: {score}, threshold: {threshold}")
        
        if score >= threshold:
            return "accept"
        else:
            return "reject"

    def get_graph(self):
        """Display the graph structure."""
        try:
            display(Image(self.graph.get_graph().print_ascii()))
        except Exception as e:
            logger.error(f"Could not display graph: {e}")
            return self.graph