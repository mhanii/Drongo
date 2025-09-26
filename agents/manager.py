from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt, Command
from langgraph.graph.message import  add_messages # Ensure this is correctly used or messages are plain lists
from typing import Annotated, List, Optional,Dict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from context.pointers import ImagePointer,DocumentPointer
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from context.store import ContextStore
from typing_extensions import TypedDict
import sqlite3
import base64
import datetime
import json
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from agents.sub_agents.find_location import FindLocationTool
from agents.tools.enums import ApplyType
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
        self.content_agent = content_agent if content_agent is not None else ContentAgent(model=model,checkpoint_path=checkpoint_path, queue=queue)
        self.last_prompt = last_prompt
        self.find_location_tool = FindLocationTool(model=self.model)
        self.document_structure = ""  # Will be set in handle_and_save_input
        self.agent = create_react_agent(
            model=self.model,
            tools=[self.generate_content, self.apply_tool_func, self.read_document],
            debug=False,
            checkpointer=MemorySaver(),
            prompt=self.get_prompt()
        )
        self.queue = queue
    
    def get_prompt(self):
        return """**You are the Manager Agent, a specialized AI orchestrator within a document editing application. Your single purpose is to translate user requests into a sequence of precise tool calls. You do not write or edit content directly.**

**Your Environment:**

*   You operate on a document represented by its `document_structure` (an HTML string). This is your ONLY map of the document.
*   User requests may include text, images, or data from uploaded documents as context.

**Your Core Workflow (MANDATORY):**

You MUST follow this sequence. Do not deviate.

1.  **ANALYZE:** First, analyze the user's request. Is the user asking to ADD new content, DELETE existing content, or EDIT existing content?
1.5. READ DOCUMENT (Optional):
    If you think you need more context you could run the read_doument tool.
2.  **STEP 1: GENERATE (Only if new content is needed):**
    *   If the request requires creating new text (e.g., "add a paragraph," "summarize this," "rewrite the title"), you **MUST** call the `generate_content` tool first.
    *   This tool will return a list of content "chunks".
        ** DON'T RETRY TO GENERATE UNLESS THE USER ASKED FOR OR IF THE GENERATION FAILED BECAUSE OF A FIXABLE ERROR **

3.  **STEP 2: APPLY (Always the final action):**
    *   To perform any modification on the document, you **MUST** call the `apply_tool_func` tool. This is your only way to change the document.
    *   You must provide the correct parameters based on your analysis.
    *   You Don't need to infer the location, it will be decided by the tool.
        ** DELETE type doesn't need a chunk_id.

**Your Tools:**

---
`read_document(asHTML: bool = False)`
    retrieves the document for you

---
`generate_content(description: str, style_guidelines: str)` SHOULD ONLY BE USED IF NECESSARY.
*   **Purpose:** Delegates all content creation to a specialized writing agent.
*   **Use When:** The user asks to write, create, generate, summarize, or add any new text.
*   **Output:** Returns a list of one or more generated content chunks. **The key for the identifier is `chunk_id`**.
    *   Example Output: `[{"chunk_id": "chunk-xyz-123", "content": "..."}]`. You **MUST** use the `chunk_id` from this output.

---

`apply_tool_func(action_type: str,  chunk_id: Optional[str] = None)`
*   **Description:** This tool finds the correct location in the document and requests the change. It can insert, delete, or edit content. Use it to make all document modifications. You must specify the action type, and for inserts/edits, the chunk_id of the content to use.
*   **Purpose:** Executes a specific change on the document structure.
*   **Parameters:**
    *   `action_type` (str): **REQUIRED.** Must be one of the following exact strings:
        *   `"INSERT"`: To add a new content chunk.
        *   `"DELETE"`: To remove an existing element. No need to provide a chunk_id.
        *   `"EDIT"`: To replace an existing element with a new content chunk.
"""

    def read_document(self,asHTML: bool = False):
        """
        Retrieves the document as text if ran with asHTML = False, or the document as html if otherwise.
        Args:
            asHTML (bool) : Wether it should be in HTML format. Text otherwise.

        Returns:
            str : (document as plain text or html string)
        """
        interrupt_payload = {
            "type": "tool_call",
            "tool_name":"read_document",
            "as_html": asHTML,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Reading document"
        }
        response = interrupt(interrupt_payload)

 
        # Extract content from response
        if isinstance(response, dict):
            return response.get("content", f"No content received from client")
        else:
            return str(response)

    def generate_content(self, description: str, style_guidelines: str = ""):
        """
        Constructs a prompt and calls the content agent's main run function. Returns a list of generated chunks with their ids.
        """
        prompt = f"Generate HTML content with the following description: {description}\nStyle guidelines: {style_guidelines}\nDocument structure: {self.document_structure}"
        
        # Assume self.content_agent.generated_chunks is a list of dicts (from chunk.to_dict())
        return self.content_agent.run(prompt)

    def apply_tool_func(self, action_type: str, chunk_id: str = ""):
        """
        Calls the FindLocationTool to decide where/how to apply a chunk and then interrupts
        to allow the client to perform the action.

        Args:
            action_type (str): The action type, one of 'INSERT', 'DELETE', or 'EDIT'.
            chunk_id (str): The chunk_id of the content to insert or edit (required for INSERT and EDIT).

        Returns:
            dict: A status message for the ReAct agent. The actual action is sent via an interrupt.
        """
        logger.info(f"--- Finding Location for Apply ---")
        logger.info(f"Type: {action_type}")
        logger.info(f"Chunk ID: {chunk_id}")

        if self.queue:
            self.queue.put_nowait(json.dumps({"type":"status_update","process":"FIND_LOCATION","status":"begin", "chunk_id": chunk_id,"apply_type":action_type}))

        try:
            apply_type_enum = ApplyType[action_type.upper()]
        except KeyError:
            logger.error(f"Invalid apply type: {action_type}")
            return {"status": "error", "message": f"Invalid apply type: {action_type}"}

        chunk_html = ""
        if apply_type_enum in [ApplyType.INSERT, ApplyType.EDIT]:
            if not chunk_id:
                return {"status": "error", "message": "chunk_id is required for INSERT or EDIT actions."}
            chunk = self.content_agent.chunk_db.load_content_chunk(chunk_id)
            if not chunk:
                logger.error(f"Chunk with id {chunk_id} not found.")
                return {"status": "error", "message": f"Chunk with id {chunk_id} not found."}
            chunk_html = chunk.html
            logger.info(f"Loaded chunk HTML with length: {len(chunk_html)}")

        result = self.find_location_tool.run(
            apply_type=apply_type_enum,
            document_structure=self.document_structure,
            last_prompt=self.last_prompt,
            chunk_id=chunk_id,
            chunk_html=chunk_html
        )

        logger.info(f"--- Find Location Finished ---")
        logger.info(f"Result: {result}")

        if result.get("status") != "success":
            return {"status": "error", "message": "Failed to find location for the requested action."}

        if self.queue:
            self.queue.put_nowait(json.dumps({"type":"status_update","process":"FIND_LOCATION","status":"end", "chunk_id": chunk_id,"apply_type":action_type}))

        interrupt_payload = {
            "type": "tool_call",
            "tool_name": "apply",  # Client expects this tool name
            "action": action_type,
            "data": {
                "chunk_html": chunk_html,
                "chunk_id": chunk_id,
                "data_position_id_start": result.get("data_position_id_start"),
                "data_position_id_end": result.get("data_position_id_end"),
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Applying change"
        }
        interrupt(interrupt_payload)

        return {"status": "success", "message": "Apply request sent to client."}
    
    def handle_and_save_input(self, request_data: dict):
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
    
    async def run_prompt(self, request_data: dict):
        payload = self.handle_and_save_input(request_data)
        logger.info(f"Manager running with payload: {payload}")
        async for chunk in self.agent.astream(payload, {"configurable": {"thread_id": "2"}}, stream_mode=["updates", "messages"]):
            data = self._handle_chunk(chunk)
            logger.info(f"Manager chunk: {data}")
            if data:
                yield data


    async def resume_agent(self,data:dict):
        logger.info(f"Manager resuming with data: {data}")
        async for chunk in self.agent.astream(Command(resume={"content": data.get('content','')}),{"configurable": {"thread_id": "2"}},stream_mode=["updates", "messages"]):
            data = self._handle_chunk(chunk)
            if data:
                yield data

    def _handle_chunk(self, chunk):
        data = None
        if (
            isinstance(chunk, tuple)
            and chunk[0] == "updates"
            and isinstance(chunk[1], dict)
            and "__interrupt__" in chunk[1]
        ):
            data = {
                "type": "interrupt",
                "payload": chunk[1]["__interrupt__"][0].value
            }
            logger.info(f"Interrupt: {data}")

        elif (
            isinstance(chunk, tuple)
            and chunk[0] == "messages"
        ):
            message = chunk[1][0]  # Extract the message content
            if message.__class__.__name__ == "AIMessageChunk" and message.content != '':
                data = {
                    "type": "message",
                    "payload": message
                }
                logger.info(f"Message: {data}")
            else:
                logger.debug(f"Ignoring non-AIMessageChunk message: {message}")
        return data