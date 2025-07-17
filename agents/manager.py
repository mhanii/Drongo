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
    def __init__(self, checkpoint_path: str, model: str, store: ContextStore, content_agent : ContentAgent=None, last_prompt: str = "", state: Optional[Dict] = State, queue=None) -> None:
        logger.info("Manager agent initialized.")
        self.connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.model = ChatGoogleGenerativeAI(model=model)
        self.CS = store if store is not None else ContextStore()
        self.content_agent = content_agent if content_agent is not None else ContentAgent(model=model,checkpoint_path=checkpoint_path, queue=queue)  # Should be passed in or set after init. The checkpoint location should change if we will use database checkpoints instead of memory based.
        self.last_prompt = last_prompt
        self.apply_tool = ApplyTool(self.content_agent.chunk_db, queue=queue)
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

    def generate_content(self, description: str, style_guidelines: str = ""):
        """
        Constructs a prompt and calls the content agent's main run function. Returns a list of generated chunks with their ids.
        """
        prompt = f"Generate HTML content with the following description: {description}\nStyle guidelines: {style_guidelines}\nDocument structure: {self.document_structure}"
        
        # Assume self.content_agent.generated_chunks is a list of dicts (from chunk.to_dict())
        return self.content_agent.run(prompt)

    def apply_tool_func(self, chunk_id: str, type: str, target_location: str = None, relative_position: str = None):
        """
        Calls the ApplyTool to decide where/how to apply a chunk. Validates the result and returns status and location info.

        Args:
            chunk_id (str): The chunk_id of the content to insert or edit (required for INSERT and EDIT).
            type (str): The action type, one of 'INSERT', 'DELETE', or 'EDIT'.


        Returns:
            dict: {"status": "success", ...} with position_id and relative_position if valid, otherwise an error dict.
        """
        result = self.apply_tool.apply(chunk_id, type, self.document_structure, self.last_prompt)
        # If error from apply_tool, propagate
        if "error" in result:
            return {"status": "error", "message": result["error"], "raw_response": result.get("raw_response")}
        # Validate presence of position_id
        position_id = result.get("position_id")
        rel_pos = result.get("relative_position")
        if not position_id:
            return {"status": "error", "message": "No position_id returned by apply_tool"}
        if type.upper() == "INSERT":
            if not rel_pos or rel_pos.upper() not in ("BEFORE", "AFTER"):
                return {"status": "error", "message": "No valid relative_position returned by apply_tool for INSERT"}
            return {"status": "success", "position_id": position_id, "relative_position": rel_pos}
        else:
            return {"status": "success", "position_id": position_id}
    

    
    def handle_and_save_input(self, request_data: dict):
        # if it has prompt, if it has images if it has docs etc
        prompt = request_data.get("text","Prompt not found")
        images = request_data.get("images",[])
        docs = request_data.get("documents",[])
        document_structure = request_data.get("document_structure", "")
        self.document_structure = document_structure
        self.last_prompt = prompt
        payload = {}

        if images:
            for image in images:
                image_pointer = ImagePointer(
                    data=image["content"],
                    filename=image["name"]
                    )
                self.CS.image_mgr.add(image_pointer)

        if docs:
            for doc in docs:
                doc_pointer = DocumentPointer(
                    data=doc["content"],
                    filename=doc["name"]
                    )
                self.CS.doc_mgr.add(doc_pointer)

        serialized_imgs = [
            {   "type": "image",
                "source_type" : "base64",
                "mime_type" : "image/jpeg",
                "filename": item.get_filename(),
                "data": item.get_data()
            }
            for item in self.CS.image_mgr.recent()
        ]
        serialized_docs = []

        for item in self.CS.doc_mgr.recent():
            pdf_bytes = base64.b64decode(item.get_data())
            file_like = BytesIO(pdf_bytes)

            elements = partition_pdf(file=file_like)

            text = "\n\n".join(el.text for el in elements)
            
            serialized_doc = {
                "type" : "text",
                "text" : text
            }
            serialized_docs.append(serialized_doc)

        doc_str = {
                "type":"text",
                "text" : f"document_structure:{self.document_structure}"
            }
        
        content = [{"type": "text", "text": prompt},doc_str] + serialized_imgs + serialized_docs 

        payload["messages"] = [{
                "role":"user",
                "content": content
            }]
        payload["document_structure"] = self.document_structure
        return payload
    
    def run_prompt(self, request_data: dict):
        payload = self.handle_and_save_input(request_data)
        response = self.agent.invoke(payload, {"configurable": {"thread_id": "2"}})
        return response['messages'][-1]


    # Remove the old apply_tool method; use self.apply_tool.apply instead

# agent = ManagerAgent("checkpoint.sqlite","models/gemini-2.0-flash",ContextStore("1",[],[]))