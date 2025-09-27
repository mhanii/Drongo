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
from langgraph.types import interrupt
import datetime

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
    data_position_id_start: str
    data_position_id_end: str
    # Retry management
    current_retry_count: int
    max_retries_reached: bool

class FindLocationTool:
    """
    A tool that determines the location to apply a change in the document.
    It does not perform the change itself, but returns the location to the caller.
    """
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
                "location_decider": "location_decider",
                "handle_error": "handle_error"
            }
        )

        graph_builder.add_conditional_edges(
            "location_decider",
            self.check_outcome,
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

    def get_prompt(self, apply_type: ApplyType, document_structure: str,  last_prompt: str, chunk_html: str = "", hint: str = "") -> str:
        """
        Generate the LLM prompt based on the apply type.
        """
        base_prompt = f"""
You are an expert document editing assistant.
### Current Document Structure
Here is the document, which is a sequence of HTML elements. Each element has a unique `data-position-id`.

{document_structure}

**User's Goal**
The user's original request was: "{last_prompt}"

### Additional Hints
{hint}

CRITICAL INSTRUCTIONS:
- You must return ONLY a single, valid JSON object and nothing else.
"""

        if apply_type == ApplyType.INSERT:
            return base_prompt + f"""
Your task is to determine the precise location to insert a new HTML chunk into an existing document.

New Content to Insert:
{chunk_html}

If the user wants to add to the end: You MUST find the VERY LAST element in the document that has a data-position-id attribute. Use that element's ID for your `data-position-id-start` and `data-position-id-end`.
If the user specifies another position: Use the data-position-id they are referring to for both `data-position-id-start` and `data-position-id-end`.

Task: Decide the best position to insert this chunk.
Return a JSON object with:
- data-position-id-start: the data-position-id of the element to insert after.
- data-position-id-end: the same as data-position-id-start for insertion.
"""
        elif apply_type == ApplyType.DELETE:
            return base_prompt + f"""
Task: Decide which data-position-id range should be deleted from the document.
Return a JSON object with:
- data-position-id-start: the data-position-id from where to begin the delete.
- data-position-id-end: the data-position-id where the deletion ends.
"""
        elif apply_type == ApplyType.EDIT:
            return base_prompt + f"""
Your task is to determine the precise location in the document to REPLACE with a new HTML chunk.

New Content to Use for Replacement:
{chunk_html}

- You must identify the element or range of elements whose content should be replaced with the new chunk.
- If the user specifies a position, use the data-position-id they are referring to.
- If the user describes the content to edit, find the best matching element(s).

Task: Decide the best position to REPLACE with this chunk.
Return a JSON object with:
- data-position-id-start: the data-position-id of the element to start the replacement from.
- data-position-id-end: the data-position-id of the element to end the replacement at.
"""
        else:
            return f"Unknown apply type: {apply_type}"

    def location_decider_action(self, state: State) -> dict:
        """Decide where to apply a chunk in the document structure using the LLM."""
        logger.info("--- Location Decider Action ---")
        logger.debug(f"--- apply_type: {state['apply_type']} ---")
        try:
            prompt = self.get_prompt(
                state["apply_type"],
                state["document_structure"],
                state["last_prompt"],
                state.get("chunk_html", ""),
                state.get("hint","")
            )
            response = self.model.invoke(prompt)
            content = response.content.strip()
            logger.info(f"Content generated by find location tool call: {content}")
            
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
                    "data_position_id_start": result.get("data-position-id-start", "-1"),
                    "data_position_id_end": result.get("data-position-id-end", "-1")
                }
            except json.JSONDecodeError as e: # Catch specific error
                logger.error(f"Error parsing find location response from LLM: {e}")
                logger.debug(f"Raw unparsable content: {content}")
                return {"outcome": "error", "result": {"error": f"Failed to parse LLM response: {e}", "raw_response": content}}

        except Exception as e:
            logger.error(f"Location decider error: {e}")
            return {"outcome": "error", "result": {"error": f"Failed to handle LLM call: {e}"}}

    def handle_error_action(self, state: State) -> dict:
        """Handle errors."""
        logger.warning("--- Handle Error Action ---")
        return {
            "result": {"error": "An error occurred during the find location process."},
            "messages": add_messages(
                state.get("messages", []),
                [{"role": "assistant", "content": "Error: Find location process failed."}]
            )
        }

    # Condition functions
    def decide_moderator_next_step(self, state: State) -> str:
        if state.get("error"):
            return "handle_error"
        if state.get("max_retries_reached", False):
            return "handle_error"
        return "location_decider"

    def check_outcome(self, state: State) -> str:
        outcome = state.get("outcome")
        logger.info(f"Location decider outcome: {outcome}")
        return outcome

    def run(self, apply_type: ApplyType, document_structure: str, last_prompt: str, chunk_id: str = "", chunk_html: str = ""):
        """
        Main entry point for the find location graph.
        """
        logger.info(f"--- Running Find Location Tool ---")
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
            "data_position_id_start": "-1",
            "data_position_id_end": "-1",
            "current_retry_count": 0,
            "max_retries_reached": False,
            "messages": []
        }

        config = {"recursion_limit": 15}
        custom_response = {}

        try:
            response = self.graph.invoke(initial_state, config)
            if self.debug:
                logger.debug(f"Find Location Graph Response: {response}")

            logger.info(f"Unparsed response:{response}")

            custom_response = {
                "status" : "error" if response.get("outcome") != "success" else "success" ,
                "data_position_id_start" : response.get("data_position_id_start","-1"),
                "data_position_id_end" : response.get("data_position_id_end","-1")
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred during graph execution: {e}")
            custom_response = {
                "status": "error",
                "message": f"An unexpected error occurred: {e}"
            }

        logger.info(f"--- Find Location Tool Finished ---")
        logger.info(f"Final Response: {custom_response}")

        return custom_response