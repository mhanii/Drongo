import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import os
from typing import Annotated, List, Optional
from time import sleep
from new_logger import get_logger
from agents.tools.enums import ApplyType

# Assuming new_logger and enums are set up correctly
logger = get_logger(True)

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    apply_type: ApplyType
    document_structure: str
    last_prompt: str
    chunk_id: str
    chunk_html: str

    # outcome
    outcome: str
    position_id: str
    relative_position: str
    # Retry management
    current_retry_count: int
    max_retries_reached: bool

class ApplyAgent:
    def __init__(self, model: ChatGoogleGenerativeAI, max_retries=3, debug=True):
        self.model = model
        self.debug = debug
        self.max_retries = max_retries

        # Build the graph
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("moderator", self.moderator_action)
        graph_builder.add_node("location_decider", self.location_decider_action)
        graph_builder.add_node("handle_error", self.handle_error_action)

        # Define edges
        graph_builder.add_edge(START, "moderator")

        # Conditional edges
        graph_builder.add_conditional_edges(
            "moderator",
            self.decide_moderator_next_step,
            {
                "decide_location": "location_decider",
                "handle_error": "handle_error"
            }
        )

        graph_builder.add_conditional_edges(
            "location_decider",
            self.check_location_decider_outcome,
            {
                "success": END,
                "error": "moderator"
            }
        )

        graph_builder.add_edge("handle_error", END)

        self.graph = graph_builder.compile()

    def moderator_action(self, state: State) -> dict:
        """Moderator validates the input and manages the flow."""
        logger.info("--- Moderator Action ---")
        apply_type = state.get("apply_type")

        if not isinstance(apply_type, ApplyType):
            logger.error(f"Invalid apply type: {apply_type}")
            return {"error": f"Invalid apply type: {apply_type}"}

        current_retries = state.get("current_retry_count", 0)

        if current_retries > self.max_retries:
            logger.warning(f"Moderator: Max retries ({self.max_retries}) exceeded.")
            return {
                "max_retries_reached": True,
                "current_retry_count": current_retries,
            }

        updates = {
            "current_retry_count": current_retries + 1,
            "max_retries_reached": False
        }

        return updates

    def get_prompt(self, type: ApplyType, document_structure: str,  last_prompt: str, chunk_html: str = ""):
        """
        Generate the LLM prompt based on the apply type.
        """
        if type == ApplyType.INSERT:
            return f"""
You are an expert document editing assistant. Your task is to determine the precise location to insert a new HTML chunk into an existing document.

### Current Document Structure
Here is the document, which is a sequence of HTML elements. Each element has a unique `position_id`.

{document_structure}

New Content to Insert:
{chunk_html}

**User's Goal**
The user's original request was: "{last_prompt}"



CRITICAL INSTRUCTIONS:
If the user wants to add to the end: You MUST find the VERY LAST element in the document that has a position_id attribute. Use that element's ID for your position_id and set relative_position to 'AFTER'.
If the user specifies another position: Use the position_id they are referring to.
You must return ONLY a single, valid JSON object and nothing else.

Task: Decide the best position to insert this chunk.
Return a JSON object with:
- position_id: the position_id of the element to insert relative to
- relative_position: 'AFTER' or 'BEFORE'
"""
        elif type == ApplyType.DELETE:
            return f"""
You are a document editing assistant. Here is the current document structure (HTML with position_id attributes):

{document_structure}

The user's original request was: "{last_prompt}"

Task: Decide which position_id should be deleted from the document.
Return a JSON object with:
- position_id: the position_id of the element to delete
"""
        elif type == ApplyType.EDIT:
            return f"""
You are an expert document editing assistant. Your task is to determine the precise location in the document to REPLACE with a new HTML chunk.

### Current Document Structure
Here is the document, which is a sequence of HTML elements. Each element has a unique `position_id`.

{document_structure}

New Content to Use for Replacement:
{chunk_html}

**User's Goal**
The user's original request was: "{last_prompt}"


CRITICAL INSTRUCTIONS:
- You must identify the SINGLE element whose content should be replaced with the new chunk.
- If the user specifies a position, use the position_id they are referring to.
- If the user describes the content to edit, find the best matching element.
- You must return ONLY a single, valid JSON object and nothing else.

Task: Decide the best position to REPLACE with this chunk.
Return a JSON object with:
- position_id: the position_id of the element to replace
"""
        else:
            return f"Unknown apply type: {type}"

    def location_decider_action(self, state: State) -> dict:
        """Decide where to apply a chunk in the document structure using the LLM."""
        logger.info("--- Location Decider Action ---")
        logger.debug(f"--- apply_type: {state['apply_type']} ---")
        try:
            prompt = self.get_prompt(
                state["apply_type"],
                state["document_structure"],
                state["last_prompt"],
                state.get("chunk_html", "")
            )
            response = self.model.invoke(prompt)
            content = response.content.strip()
            logger.info(f"Content generated by apply tool call: {content}")
            
            # More robustly find the JSON object
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no markdown, assume the whole content is the JSON object
                json_str = content

            try:
                result = json.loads(json_str)
                logger.debug(result)
                return {
                    "outcome": "success",
                    "position_id": result.get("position_id", "-1"),
                    "relative_position": result.get("relative_position", "NONE")
                }
            except json.JSONDecodeError as e: # Catch specific error
                logger.error(f"Error parsing apply response from LLM: {e}")
                logger.debug(f"Raw unparsable content: {content}")
                return {"outcome": "error", "result": {"error": f"Failed to parse LLM response: {e}", "raw_response": content}}

        except Exception as e:
            logger.error(f"Location decider error: {e}")
            return {"outcome": "error", "result": {"error": f"Failed to handle LLM call: {e}"}}

    def handle_error_action(self, state: State) -> dict:
        """Handle errors."""
        logger.warning("--- Handle Error Action ---")
        return {
            "result": {"error": "An error occurred during the apply process."},
            "messages": add_messages(
                state.get("messages", []),
                [{"role": "assistant", "content": "Error: Apply process failed."}]
            )
        }

    # Condition functions
    def decide_moderator_next_step(self, state: State) -> str:
        if state.get("error"):
            return "handle_error"
        if state.get("max_retries_reached", False):
            return "handle_error"
        return "decide_location"

    def check_location_decider_outcome(self, state: State) -> str:
        outcome = state.get("outcome")
        logger.info(f"Location decider outcome: {outcome}")
        return outcome

    def run(self, apply_type: ApplyType, document_structure: str, last_prompt: str, chunk_id: str = "", chunk_html: str = ""):
        """
        Main entry point for the apply graph.
        """
        logger.info(f"--- Running Apply Agent ---")
        logger.info(f"Apply Type: {apply_type}")
        logger.info(f"Document Structure Length: {len(document_structure)}")
        logger.info(f"Last Prompt: {last_prompt}")
        logger.info(f"Chunk ID: {chunk_id}")
        logger.info(f"Chunk HTML Length: {len(chunk_html)}")

        initial_state = {
            "apply_type": apply_type,
            "document_structure": document_structure,
            "last_prompt": last_prompt,
            "chunk_id": chunk_id,
            "chunk_html": chunk_html,

            "outcome":"N/A",
            "position_id": "-1", # FIX: Changed to string for consistency
            "relative_position":"NONE",
            "current_retry_count": 0,
            "max_retries_reached": False,
            "messages": []
        }

        config = {"recursion_limit": 15}
        custom_response = {}

        try:
            response = self.graph.invoke(initial_state, config)
            if self.debug:
                logger.debug(f"Apply Graph Response: {response}")

            logger.info(f"Unparsed response:{response}")

            custom_response = {
                "status" : "error" if response.get("outcome") != "success" else "success" ,
                "position_id" : response.get("position_id","-1"),
                "relative_position" : response.get("relative_position","NONE")
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred during graph execution: {e}")
            custom_response = {
                "status": "error",
                "message": f"An unexpected error occurred: {e}"
            }

        logger.info(f"--- Apply Agent Finished ---")
        logger.info(f"Final Response: {custom_response}")

        return custom_response