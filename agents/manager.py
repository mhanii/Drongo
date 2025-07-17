from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import  add_messages # Ensure this is correctly used or messages are plain lists
from typing import Annotated, List, Optional,Dict
from langgraph.checkpoint.sqlite import SqliteSaver
from context.pointers import ImagePointer,DocumentPointer
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from context.store import ContextStore
from typing_extensions import TypedDict
import sqlite3
import base64
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from agents.tools.apply import ApplyTool
from agents.content import ContentAgent
from new_logger import get_logger

logger = get_logger()


class State(TypedDict):
    messages : Annotated[List,add_messages] 


class ManagerAgent:
    def __init__(self, checkpoint_path: str, model: str, store: ContextStore, content_agent : ContentAgent=None, last_prompt: str = "", state: Optional[Dict] = State) -> None:
        logger.info("Manager agent initialized.")
        self.connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.model = ChatGoogleGenerativeAI(model=model)
        self.CS = store if store is not None else ContextStore()
        self.content_agent = content_agent if content_agent is not None else ContentAgent(model=model,checkpoint_path=checkpoint_path)  # Should be passed in or set after init. The checkpoint location should change if we will use database checkpoints instead of memory based.
        self.last_prompt = last_prompt
        self.apply_tool = ApplyTool(self.content_agent.chunk_db)
        self.document_structure = ""  # Will be set in handle_and_save_input
        self.agent = create_react_agent(
            model=self.model,
            tools=[self.generate_content, self.apply_tool_func],
            debug=True,
            checkpointer=SqliteSaver(self.connection),
            prompt=self.get_prompt()
        )

    
    def get_prompt(self):
        return """**You are the Manager Agent, a specialized AI orchestrator within a document editing application. Your single purpose is to translate user requests into a sequence of precise tool calls. You do not write or edit content directly.**

**Your Environment:**

*   You operate on a document represented by its `document_structure` (an HTML string). This is your ONLY map of the document.
*   User requests may include text, images, or data from uploaded documents as context.

**Your Core Workflow (MANDATORY):**

You MUST follow this sequence. Do not deviate.

1.  **ANALYZE:** First, analyze the user's request. Is the user asking to ADD new content, DELETE existing content, or EDIT existing content?

2.  **STEP 1: GENERATE (Only if new content is needed):**
    *   If the request requires creating new text (e.g., "add a paragraph," "summarize this," "rewrite the title"), you **MUST** call the `generate_content` tool first.
    *   This tool will return a list of content "chunks".

3.  **STEP 2: APPLY (Always the final action):**
    *   To perform any modification on the document, you **MUST** call the `apply_tool_func` tool. This is your only way to change the document.
    *   You must provide the correct parameters based on your analysis.

**Your Tools (Revised Definitions):**

---

`generate_content(description: str, style_guidelines: str)`
*   **Purpose:** Delegates all content creation to a specialized writing agent.
*   **Use When:** The user asks to write, create, generate, summarize, or add any new text.
*   **Output:** Returns a list of one or more generated content chunks. **The key for the identifier is `chunk_id`**.
    *   Example Output: `[{"chunk_id": "chunk-xyz-123", "content": "..."}]`. You **MUST** use the `chunk_id` from this output.

---

`apply_tool_func(action_type: str, target_location: str, chunk_id: Optional[str] = None, relative_position: Optional[str] = None)`
*   **Description:** This tool applies a structural change to the document. It can insert, delete, or edit content at a specified location. Use it to make all document modifications. You must specify the action type, the target location (by selector or position_id), and, for inserts/edits, the chunk_id of the content to use. For inserts, also specify whether to insert BEFORE or AFTER the target.
*   **Purpose:** Executes a specific change on the document structure.
*   **Parameters:**
    *   `action_type` (str): **REQUIRED.** Must be one of the following exact strings:
        *   `"INSERT"`: To add a new content chunk.
        *   `"DELETE"`: To remove an existing element.
        *   `"EDIT"`: To replace an existing element with a new content chunk.

    *   `target_location` (str): **REQUIRED.** A precise CSS selector or description identifying the target HTML element(s).
        *   For `INSERT`, this is the *anchor element* for the insertion.
        *   For `DELETE` and `EDIT`, this is the element to be removed or replaced.

"""