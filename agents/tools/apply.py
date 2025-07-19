import json
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

from database.content_chunk_db import ContentChunkDB
from agents.sub_agents.apply import ApplyAgent
from agents.tools.enums import ApplyType

from new_logger import get_logger

logger = get_logger()

class ApplyTool:
    def __init__(self, content_db : ContentChunkDB, model: str = "models/gemini-2.5-pro",  queue: asyncio.Queue = None):
        self.model = ChatGoogleGenerativeAI(model=model)
        self.content_db = content_db
        self.queue = queue
        self.apply_agent = ApplyAgent(model=self.model)

    def apply(self, type: str, chunk_id: str,  document_structure: str, last_prompt: str):
        """
        Decide where to apply a chunk in the document structure using the LLM.
        For type 'INSERT', returns position_id and relative_position ('AFTER' or 'BEFORE').
        """
        logger.info(f"--- Applying Chunk ---")
        logger.info(f"Type: {type}")
        logger.info(f"Chunk ID: {chunk_id}")
        logger.info(f"Document Structure Length: {len(document_structure)}")
        logger.info(f"Last Prompt: {last_prompt}")

        if self.queue:
            self.queue.put_nowait(json.dumps({"status": "applying_chunk", "chunk_id": chunk_id, "type": type}))

        try:
            apply_type = ApplyType[type.upper()]
        except KeyError:
            logger.error(f"Invalid apply type: {type}")
            return {"error": f"Invalid apply type: {type}"}

        chunk_html = ""
        if apply_type == ApplyType.INSERT or apply_type == ApplyType.EDIT:
            chunk = self.content_db.load_content_chunk(chunk_id)
            if not chunk:
                logger.error(f"Chunk with id {chunk_id} not found.")
                return {"error": f"Chunk with id {chunk_id} not found."}
            chunk_html = chunk.html
            logger.info(f"Loaded chunk HTML with length: {len(chunk_html)}")

        result = self.apply_agent.run(
            apply_type=apply_type,
            document_structure=document_structure,
            last_prompt=last_prompt,
            chunk_id=chunk_id,
            chunk_html=chunk_html
        )

        logger.info(f"--- Apply Tool Finished ---")
        logger.info(f"Result: {result}")
        return result