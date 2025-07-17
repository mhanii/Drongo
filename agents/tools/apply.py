import json
from langchain_google_genai import ChatGoogleGenerativeAI

from database.content_chunk_db import ContentChunkDB
class ApplyTool:
    def __init__(self, content_db : ContentChunkDB, model: str = "models/gemini-2.5-pro", websocket=None):
        self.model = ChatGoogleGenerativeAI(model=model)
        self.content_db = content_db
        self.websocket = websocket

    def get_prompt(self, type: str, document_structure: str, chunk_html: str, last_prompt: str):
        """
        Generate the LLM prompt based on the apply type.
        """
        if type.upper() == "INSERT":
            return f"""
You are an expert document editing assistant. Your task is to determine the precise location to insert a new HTML chunk into an existing document.

### Current Document Structure
Here is the document, which is a sequence of HTML elements. Each element has a unique `position_id`.

{document_structure}

New Content to Insert:
{chunk_html}

**User's Goal**
The user's original request was: "{last_prompt}"
{last_prompt}


CRITICAL INSTRUCTIONS:
If the user wants to add to the end: You MUST find the VERY LAST element in the document that has a position_id attribute. Use that element's ID for your position_id and set relative_position to 'AFTER'.
If the user specifies another position: Use the position_id they are referring to.
You must return ONLY a single, valid JSON object and nothing else.

Task: Decide the best position to insert this chunk. 
Return a JSON object with:
- position_id: the position_id of the element to insert relative to
- relative_position: 'AFTER' or 'BEFORE'
"""
        elif type.upper() == "DELETE":
            return f"""
You are a document editing assistant. Here is the current document structure (HTML with position_id attributes):

{document_structure}

Task: Decide which position_id should be deleted from the document. 
Return a JSON object with:
- position_id: the position_id of the element to delete
"""
        elif type.upper() == "EDIT":
            return f"""
You are an expert document editing assistant. Your task is to determine the precise location in the document to REPLACE with a new HTML chunk.

### Current Document Structure
Here is the document, which is a sequence of HTML elements. Each element has a unique `position_id`.

{document_structure}

New Content to Use for Replacement:
{chunk_html}

**User's Goal**
The user's original request was: "{last_prompt}"
{last_prompt}


CRITICAL INSTRUCTIONS:
- You must identify the SINGLE element whose content should be replaced with the new chunk.
- If the user specifies a position, use the position_id they are referring to.
- If the user describes the content to edit, find the best matching element.
- You must return ONLY a single, valid JSON object and nothing else.

Task: Decide the best position to REPLACE with this chunk.
Return a JSON object with:
- position_id: the position_id of the element to replace
"""
        # Extend for other types as needed
        else:
            return f"Unknown apply type: {type}"

    def apply(self, chunk_id: str, type: str, document_structure: str, last_prompt: str):
        """
        Decide where to apply a chunk in the document structure using the LLM.
        For type 'INSERT', returns position_id and relative_position ('AFTER' or 'BEFORE').
        """
        if self.websocket:
            self.websocket.send(json.dumps({"status": "applying_chunk", "chunk_id": chunk_id, "type": type}))
        chunk_html = ""
        if type.upper() == "INSERT" or type.upper() == "EDIT":
            chunk = self.content_db.load_content_chunk(chunk_id)
            if not chunk:
                return {"error": f"Chunk with id {chunk_id} not found."}
            chunk_html = chunk.html
        prompt = self.get_prompt(type, document_structure, chunk_html, last_prompt)
        response = self.model.invoke(prompt)
        try:
            # Strip Markdown code block markers if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.startswith("```"):
                content = content[len("```"):].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
            result = json.loads(content)
        except Exception as e:
            result = {"error": f"Failed to parse LLM response: {str(e)}", "raw_response": response.content}
        return result 