from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages # Ensure this is correctly used or messages are plain lists
from pydantic import BaseModel
from utils.html_validator import HTMLValidator # Assuming this import is correct
import os
from typing import Annotated, List
from IPython.display import Image, display

# Define State with potentially missing fields if not always present
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    html: str
    description: str
    style_guidelines: str
    formatted_context_history: str

    content_generation_outcome: str
    validation_outcome: str
    current_prompt: str
    evaluator_score: int
    evaluator_feedback: str

    # Add these for robust state management
    current_retry_count: int
    best_html_so_far: str
    best_score_so_far: int
    max_retries_reached: bool # Flag to signal exiting due to retries

class EvaluatorResponse(BaseModel):
    score: int
    feedback: str

class HtmlAgent:
    def __init__(self, model="models/gemini-1.5-flash-latest", disallowed_tags=["script", "iframe", "style", "link", "meta", "head", "body", "html", "div", "em", "br", "i", "b"], allowed_tags=["p", "span", "u", "ol", "ul", "li", "table", "tr", "td", "th", "tbody", "h1", "h2", "h3", "h4", "h5", "h6"], acceptance_threshold=96, max_retries=3): # Updated model name if necessary
        self.moderator_lm = ChatGoogleGenerativeAI(model=model)
        self.content_lm = ChatGoogleGenerativeAI(model=model)
        self.evaluator_lm = ChatGoogleGenerativeAI(model=model)
        self.html_validator = HTMLValidator()

        graph_builder = StateGraph(State)
        self.memory = MemorySaver()
    
        self.config = {"configurable": {"thread_id": "1"}}

        self.disallowed_tags = disallowed_tags
        self.allowed_tags = allowed_tags
        self.acceptance_threshold = acceptance_threshold

        self.max_retries = max_retries
        self.best_html = ""
        self.best_score = 0
        self.retries = 0
        # Define Nodes with proper actions
        graph_builder.add_node("moderator", self.moderator_action)
        graph_builder.add_node("content_generator", self.content_generator_action)
        graph_builder.add_node("html_validator_node", self.html_validator_action)
        graph_builder.add_node("evaluator", self.evaluator_action)
        graph_builder.add_node("handle_content_error", self.handle_content_error_action)

        # Define Edges
        graph_builder.add_edge(START, "moderator")
        graph_builder.add_edge("moderator", "content_generator")

        # Conditional edge from content_generator
        graph_builder.add_conditional_edges(
            "content_generator",
            self.check_content_generation_outcome,
            {
                "success": "html_validator_node",
                "error": "handle_content_error", # Route to error handler
            }
        )
        # Edge from error handler back to moderator (or END, or another recovery step)
        graph_builder.add_edge("handle_content_error", "moderator") # Example: retry by going to moderator

        # Conditional edge from html_validator_node
        graph_builder.add_conditional_edges(
            "html_validator_node",
            self.check_validation_outcome,
            {
                "success": "evaluator",
                "error": "moderator", # Or "handle_content_error" or a specific validation error node
            }
        )
        
        # Conditional edge from evaluator based on score
        graph_builder.add_conditional_edges(
            "evaluator",
            self.check_evaluation_score,
            {
                "accept": END, # Loop back for another round if accepted for refinement
                "reject": "moderator",         # End if rejected
            }
        )

        graph_builder.add_conditional_edges(
            "moderator",
            self.decide_moderator_next_step,
            {
                "generate_content": "content_generator",
                "end_with_best": "prepare_final_output" # New node to set final HTML from best_html_so_far
            }
        )
        graph_builder.add_node("prepare_final_output", self.prepare_final_output_action)
        graph_builder.add_edge("prepare_final_output", END)

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def run(self, description: str, style_guidelines: str, formatted_context_history: str):
    # Reset agent-level retry count IF it's not part of the state (but it should be)
    # self.retries = 0 # If keeping as instance var, reset here. Better to move to State.

        initial_state = {
            "messages": [{"role": "system", "content": self._get_content_generation_prompt_string()},
                        {"role": "user", "content": f"Initial task: Description: {description}\nStyle Guidelines: {style_guidelines}\nFormatted Context History: {formatted_context_history}"}], # Include context here
            "description": description,
            "style_guidelines": style_guidelines,
            "formatted_context_history": formatted_context_history,
            "html": "",
            "current_retry_count": 0, # Initialize in state
            "best_html_so_far": "",   # Initialize in state
            "best_score_so_far": -1,  # Use -1 or float('-inf') to indicate no score yet
            "max_retries_reached": False
        }
        return self.graph.invoke(initial_state, self.config)

    def moderator_action(self, state: State) -> dict:
        print("--- Moderator Action ---")
        messages = state.get("messages", [])
        current_retries = state.get("current_retry_count", 0)
        evaluator_feedback = state.get("evaluator_feedback")
        evaluator_score = state.get("evaluator_score")
        validation_outcome = state.get("validation_outcome")

        # This attempt IS a retry if evaluator_score is present (meaning we came from evaluator)
        # or if validation_outcome is error (meaning we came from validator error path)
        is_retry_loop = (evaluator_score is not None) or (validation_outcome == "error")

        if is_retry_loop:
            current_retries += 1 # This is now the count for the *upcoming* attempt

        if current_retries > self.max_retries: # Use > because current_retries is for the *next* attempt
            print(f"Moderator: Max retries ({self.max_retries}) exceeded. Current attempt would be {current_retries}.")
            return {"max_retries_reached": True, "current_retry_count": current_retries -1 } # current_retry_count is how many were completed

        updates = {
            "html": "", # Clear previous HTML for new generation
            "evaluator_score": None,
            "evaluator_feedback": None,
            "content_generation_outcome": None,
            "validation_outcome": None,
            "current_retry_count": current_retries, # Update retry count for this attempt
            "max_retries_reached": False
        }
        new_user_message_content = ""

        if evaluator_feedback and evaluator_score is not None and evaluator_score < self.acceptance_threshold:
            print(f"Moderator: Re-prompting (Attempt {current_retries}/{self.max_retries}) due to low score ({evaluator_score}). Feedback: {evaluator_feedback}")
            new_user_message_content = (
                f"The previous HTML (Attempt {current_retries-1}) was reviewed. Score: {evaluator_score}. Please try again, addressing this feedback:\n"
                f"{evaluator_feedback}\n"
                "Focus on incorporating this feedback. Adhere strictly to all HTML generation rules."
            )
        elif validation_outcome == "error": # Check this specific state field
            validation_error_details = "Previous HTML had validation issues."
            # Find the validation error message if added by the validator
            for msg in reversed(messages): # Search existing messages
                if msg["role"] == "ai" and "Validation Error:" in msg.get("content", ""): # Or tool
                    validation_error_details = msg.get("content")
                    break
            print(f"Moderator: Re-prompting (Attempt {current_retries}/{self.max_retries}) due to HTML validation error. Details: {validation_error_details}")
            new_user_message_content = (
                f"The previous HTML (Attempt {current_retries-1}) failed validation: '{validation_error_details}'.\n"
                f"Please regenerate. Pay extreme attention to syntax, allowed tags ({self.allowed_tags}), and text encapsulation."
            )
        else: # Initial pass
            print(f"Moderator: Initial pass (Attempt {current_retries}/{self.max_retries}). Preparing for content generation.")
            # The initial user message is already in state["messages"] from run().
            # We can add an assistant message to confirm moderator action.
            updates["messages"] = add_messages(messages, [{"role": "assistant", "content": "Moderator: Request details processed. Proceeding to content generation."}])
            return updates # No new user message, use existing

        if new_user_message_content:
            updates["messages"] = add_messages(messages, [{"role": "user", "content": new_user_message_content}])

        return updates



    def _get_content_generation_prompt_string(self) -> str:
        # This is where the logic from your old `generate_content_prompt` method (the string itself) goes.
        # It should be dynamically formatted if needed.
        # CRITICAL: Ensure this prompt strongly guides the LLM to ONLY output HTML.
        base_prompt = f"""
        You are an expert content generator specializing in creating structured HTML content.
        Your task is to generate HTML based *only* on the following description, guidelines, context, and history.

        --- CONTENT GENERATION RULES ---
        {self._get_content_generation_rules()}
        --- END CONTENT GENERATION RULES ---

        --- EXAMPLE OF CORRECT HTML STRUCTURE ---
        <p style="line-height:1.38; font-family:Arial,sans-serif;"><span style="font-size:11pt;">Regular text <span style="font-weight:bold;">bold</span> and <span style="font-style:italic;">italic</span>.</span></p>

        CRITICAL INSTRUCTION: Respond with ONLY the generated HTML content. No explanations, introductions, or markdown formatting (```html...). Start directly with the first HTML tag (e.g., `<p...>` or `<h1...>`). Generate NEW content based on the 'DESCRIPTION' and 'STYLE GUIDELINES', considering the 'PROVIDED CONTEXT' and 'PREVIOUS CHAT HISTORY'. Do not repeat the context or history in your output.

        Generate the HTML content now.
        """
        # return base_prompt.format(
        #     description=state["description"],
        #     style_guidelines=state["style_guidelines"],
        #     formatted_context_history=state["formatted_context_history"] # Or construct from state["messages"]
        # )
        return base_prompt

    def _get_content_generation_rules(self) -> str:
        return f"""
        1.  **Valid Structure:** Ensure perfect HTML syntax.
        2.  **Allowed Tags ONLY:** Use ONLY {self.allowed_tags}.
        3.  **Forbidden Tags:** DO NOT use {self.disallowed_tags}.
        4.  **Text Encapsulation:** ALL visible text MUST be directly inside `<span>`.
        5.  **Span Placement:** Place `<span>` immediately inside block elements (`<p>`, `<h1-h6>`, etc.).
        6.  **Root Element:** Start with `<p>` or `<h1-h6>`. No `<!DOCTYPE>`, `<html>`, etc.
        7.  **Tables:** MUST contain `<tbody>`. Rows (`<tr>`) inside `<tbody>`. Use `<th>` for header cells.
        8.  **Colors:** Use HEX format (`#FF0000`).
        9.  **Styling:** Use inline `style` attribute ONLY. Valid `property: value;`.
        10. **Headings:** Use `h1`-`h6` appropriately.
        11. **New Lines/Spacing:** Use `<p>` tags. For empty space use `<p><span> </span></p>` or margins. DO NOT use `\\n`. Note: ` ` for non-breaking space.
        12. **Styling Logic:** Apply general styles to parent block (`<p>`, `<h1>`). Use `<span>` for inline changes ONLY (bold, italic, color).
        13. **Attribute Validity:** Ensure valid CSS (`property: value;`) in double quotes. Use standard font names.
        """
    def content_generator_action(self, state: State) -> dict:
        print("--- Content Generator Action ---")
        try:
            response = self.content_lm.invoke(state["messages"], self.config)
            generated_html = response.content # Assuming response.content holds the string
            print(f"Generated HTML: {generated_html}")
            # It's good practice to clean the LLM output slightly before validation

            return {
                "html": generated_html,
                "content_generation_outcome": "success",
                "messages": [{"role": "assistant", "content":f"Generated HTML: {generated_html}"}]
            }
        except Exception as e:
            print(f"Content generation error: {e}")
            return {
                "html": f"<p><span>Error during content generation: {e}</span></p>", # Provide some error HTML
                "content_generation_outcome": "error",
                "messages": [{"role": "assistant", "content": f"Error in content generation: {e}"}]
            }

    def html_validator_action(self, state: State) -> dict:
        print("--- HTML Validator Action ---")
        html_to_validate = state.get("html", "")
        if not html_to_validate:
            return {
                "validation_outcome": "error",
                "messages": [{"role": "assistant", "content": "Validation error: No HTML content to validate."}]
            }

        # _clean_llm_html_output was already called in content_generator_action
        # If not, call it here: cleaned_html = self.html_validator._clean_llm_html_output(html_to_validate)
        cleaned_html = self.html_validator._clean_llm_html_output(html_to_validate)
        validation_result = self.html_validator.validate_and_repair(cleaned_html) # Pass the already cleaned html

        if validation_result["status"] == "error":
            return {
                "validation_outcome": "error",
                "html": validation_result.get("html", html_to_validate), # Keep original or repaired attempt
                "messages": [{"role": "ai", "content":f"Validation Error: {validation_result.get('html', '')}"}]
            }
        
        return {
            "html": validation_result["html"],
            "validation_outcome": "success",
            "messages": [{"role": "ai", "content": f"Validation Success: {validation_result.get('html', '')}"}]
        }

    def _get_evaluator_prompt_string(self, state: State) -> str:

        
        task_prompt = f"Description: {state['description']}\nStyle Guidelines: {state['style_guidelines']}"
        html_content = state['html']
        
        return f"""
            You are an expert content evaluator.
            Your task is to evaluate the provided HTML content based on the original task requirements and general quality.
            The HTML content was generated based on the following task prompt:

            --- TASK RULES ---
            {self._get_content_generation_rules()}
            --- END TASK RULES ---

            --- TASK PROMPT ---
            {task_prompt}
            --- END TASK PROMPT ---

            --- HTML CONTENT TO EVALUATE ---
            {html_content}
            --- END HTML CONTENT ---

            
            Please provide a score from 0 to 100 (where 100 is excellent) and detailed feedback.
            Consider adherence to the task prompt, HTML validity (though it should have been pre-validated), and overall quality.
            Respond with a JSON object matching the following Pydantic model:
            {{
            "score": "integer (0-100)",
            "feedback": "string (your detailed feedback)"
            }}
        """

    def evaluator_action(self, state: State) -> dict:
        print("--- Evaluator Action ---")
        prompt_str = self._get_evaluator_prompt_string(state)
        current_html = state.get("html", "")
        best_html_so_far = state.get("best_html_so_far", "")
        best_score_so_far = state.get("best_score_so_far", -1)

        try:
            response = self.evaluator_lm.with_structured_output(EvaluatorResponse).invoke(prompt_str)
            
            new_best_html = best_html_so_far
            new_best_score = best_score_so_far

            if response.score > best_score_so_far:
                new_best_score = response.score
                new_best_html = current_html # The HTML just evaluated
                print(f"Evaluator: New best score: {new_best_score} (previous: {best_score_so_far})")
            
            return {
                "evaluator_score": response.score,
                "evaluator_feedback": response.feedback,
                "best_html_so_far": new_best_html,
                "best_score_so_far": new_best_score,
                "messages": add_messages(state.get("messages"), [{"role": "assistant", "content": f"Evaluation Feedback: {response.feedback} (Score: {response.score})"}])
            }
        except Exception as e:
            print(f"Evaluator error: {e}")
            return {
                "evaluator_score": 0,
                "evaluator_feedback": f"Error during evaluation: {e}", # Corrected f-string
                "best_html_so_far": best_html_so_far, # Keep previous best on error
                "best_score_so_far": best_score_so_far,
                "messages": add_messages(state.get("messages"), [{"role": "assistant", "content": f"Error during evaluation: {e}"}]) # Corrected f-string
            }

    def handle_content_error_action(self, state: State) -> dict:
        print("--- Handle Content Error Action ---")
        # This node is reached if content generation or validation fails severely.
        # You might log the error, set a default error HTML, etc.
        error_message = "Content generation or validation failed. Retrying or modifying approach."
        return {
            "html": "<p><span>Content processing encountered an error.</span></p>",
            "messages": [{"role": "assistant", "content": f"Error: {error_message}"}]
        }

    # --- Condition Functions ---
    def check_content_generation_outcome(self, state: State) -> str:
        print(f"--- Condition: Content Generation Outcome is {state.get('content_generation_outcome')} ---")
        return state.get("content_generation_outcome", "error")

    def check_validation_outcome(self, state: State) -> str:
        print(f"--- Condition: Validation Outcome is {state.get('validation_outcome')} ---")

        return state.get("validation_outcome", "error")

    def check_evaluation_score(self, state: State) -> str:
        score = state.get("evaluator_score", 0)
        feedback = state.get("evaluator_feedback", "")
        print(f"--- Condition: Evaluation Score is {score} --- {feedback}")
        if score >= self.acceptance_threshold:
            return "accept"
        else:
            
            return "reject"


    def decide_moderator_next_step(self, state: State) -> str:
        if state.get("max_retries_reached", False):
            return "end_with_best"
        return "generate_content"
        
    def prepare_final_output_action(self, state: State) -> dict:
        print("--- Max retries reached or error. Finalizing with best available HTML. ---")
        return {
            "html": state.get("best_html_so_far", "<p><span>No satisfactory HTML generated.</span></p>"),
            "evaluator_score": state.get("best_score_so_far", 0),
            "messages": [{"role": "assistant", "content": "Max retries reached. Using best HTML found."}]
        }
    def get_graph(self):
        try:
            display(Image(self.graph.get_graph().print_ascii()))
        except Exception as e:
            print(f"Error displaying graph: {e}")
            # This requires some extra dependencies and is optional
            pass

# Example usage (ensure GOOGLE_API_KEY is set in your environment)
if __name__ == '__main__':
    # from dotenv import load_dotenv
    # load_dotenv() # If you use .env file for API keys

    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set the GOOGLE_API_KEY environment variable.")
    else:
        agent = HtmlAgent()
        test_description = "Create a short paragraph about the benefits of hydration, then a list of 3 tips for staying hydrated."
        test_style_guidelines = "Use Arial font, 12pt. The paragraph should have normal line height. List items should be underlined."
        test_context_history = "No previous conversation."

        result = agent.run(test_description, test_style_guidelines, test_context_history)
        
        print("\n--- Final State ---")
        # print(result) # This will print the full state dict, which can be verbose
        
        print("\n--- Final Messages ---")
        if result and 'messages' in result:
            for msg in result['messages']:
                print(f"{msg['role']}: {msg['content']}")
        
        print("\n--- Final HTML ---")
        if result and 'html' in result:
            print(result['html'])

