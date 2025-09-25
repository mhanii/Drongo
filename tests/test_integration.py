import asyncio
import pytest
import websockets
import logging

HOST = "localhost"
PORT = 8765  # Change if your server uses a different port
WS_URL = f"ws://{HOST}:{PORT}/ws"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_full_project_flow():
    import json
    async with websockets.connect(WS_URL) as websocket:
        # Step 1: Handshake
        handshake_request = {"type": "handshake"}
        logger.info(f"Sending handshake request: {handshake_request}")
        await websocket.send(json.dumps(handshake_request))
        handshake_response = await websocket.recv()
        logger.info(f"Received handshake response: {handshake_response}")
        handshake_obj = json.loads(handshake_response)
        assert handshake_obj.get("status") == "created", f"Handshake failed: {handshake_obj}"
        session_id = handshake_obj.get("session_id")
        assert session_id, f"No session_id in handshake response: {handshake_obj}"

        # Step 2: Send prompt with session id
        prompt_request = {
            "type": "prompt",
            "text": "Could you use read_document tool to make sure everything works just fine?",
            "images": [],
            "documents": [],
            "document_structure": "<p></p>",
            "session_id": session_id
        }
        logger.info(f"Sending prompt request: {prompt_request}")
        await websocket.send(json.dumps(prompt_request))
        prompt_response = await websocket.recv()
        logger.info(f"Received prompt response: {prompt_response}")
        prompt_obj = json.loads(prompt_response)
        assert prompt_obj.get("type") == "interrupt", f"Prompt failed: {prompt_obj}"
        
        # Simulate tool response
        tool_response = {
            "type": "tool_response",
            "tool_name": "read_document",
            "content": "<p>this is the document</p>",
            "session_id": session_id
        }
        logger.info(f"Sending tool response: {tool_response}")
        await websocket.send(json.dumps(tool_response))
        tool_response_ack = await websocket.recv()
        logger.info(f"Received tool response ack: {tool_response_ack}")
        # # Step 3: Close connection with session id
        # close_request = {"session_id": session_id, "action": "close"}
        # logger.info(f"Sending close request: {close_request}")
        # await websocket.send(json.dumps(close_request))
        # close_response = await websocket.recv()
        # close_obj = json.loads(close_response)
        # assert close_obj.get("status") == "closed", f"Close failed: {close_obj}"
        # assert close_obj.get("session_id") == session_id, f"Session id mismatch on close: {close_obj}"

if __name__ == "__main__":
    asyncio.run(test_full_project_flow())
