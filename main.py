import asyncio
import websockets
import json
import uuid  # Import the UUID library for unique session IDs
from agents.manager import ManagerAgent
from new_logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
logger = get_logger(DEBUG)

agent_store = {}

async def handler(websocket, path):
    """Routes incoming websocket connections based on the path."""
    logger.info(f"New connection on path: {path}")
    if path == "/prompt":
        await handle_prompt(websocket)
    elif path == "/tool-message":
        await handle_tool_message(websocket)
    else:
        logger.warning(f"Connection attempt on unknown path: {path}")
        await websocket.close(code=1011, reason=f"Unknown path: {path}")

async def handle_prompt(websocket):
    """Handles the lifecycle of a prompt-response session with an agent."""
    # --- CHANGE: Use UUID for a robust, unique session ID ---
    session_id = str(uuid.uuid4())
    logger.info(f"Created new agent session: {session_id}")

    agent = ManagerAgent(
        checkpoint_path="data/manager_checkpoint.sqlite",
        model="models/gemini-2.5-pro",
        store=None,
        websocket=websocket
    )
    agent_store[session_id] = agent

    try:
        async for message in websocket:
            try:
                input_data = json.loads(message)

                # --- CHANGE: Updated and more robust input validation ---
                required_fields = {
                    "text": str,
                    "images": list,
                    "documents": list,
                    "document_structure": str
                }

                for field, field_type in required_fields.items():
                    if field not in input_data:
                        raise ValueError(f"Required field '{field}' is missing.")
                    if not isinstance(input_data[field], field_type):
                        raise ValueError(f"Field '{field}' must be of type {field_type.__name__}.")

                # If validation passes, the input_data is ready for the agent
                logger.info(f"Processing prompt for session {session_id}")
                response = agent.run_prompt(input_data)
                
                logger.info(f"Agent response for {session_id}: {response.content}")
                await websocket.send(json.dumps({
                    "message": "Prompt processed successfully.",
                    "agent_response": response.content,
                    "session_id": session_id # Good practice to return the session_id
                }))

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message.")
                await websocket.send(json.dumps({"error": "Invalid JSON format."}))
            except ValueError as ve:
                logger.error(f"Invalid input data: {ve}")
                await websocket.send(json.dumps({"error": str(ve)}))
            except Exception as e:
                logger.error(f"An unexpected error occurred in handle_prompt: {e}", exc_info=DEBUG)
                await websocket.send(json.dumps({"error": "An unexpected server error occurred."}))

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed for session {session_id} (Code: {e.code}, Reason: {e.reason})")
    finally:
        if session_id in agent_store:
            logger.info(f"Cleaning up agent for session {session_id}")
            del agent_store[session_id]

async def handle_tool_message(websocket):
    """Handles incoming messages from a tool."""
    try:
        async for message in websocket:
            try:
                tool_data = json.loads(message)
                session_id = tool_data.get("session_id")
                if not session_id or not isinstance(session_id, str):
                    await websocket.send(json.dumps({"error": "session_id is missing or invalid."}))
                    continue
                
                agent = agent_store.get(session_id)
                if not agent:
                    await websocket.send(json.dumps({"error": f"Agent not found for session_id: {session_id}."}))
                    continue

                # Placeholder for your actual logic to handle the tool's response
                # agent.handle_tool_response(tool_data)

                logger.info(f"Received tool message for session {session_id}: {tool_data}")
                await websocket.send(json.dumps({"status": "received"}))

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message.")
                await websocket.send(json.dumps({"error": "Invalid JSON format."}))
            except Exception as e:
                logger.error(f"An error occurred in handle_tool_message: {e}", exc_info=DEBUG)
                await websocket.send(json.dumps({"error": "An unexpected error occurred."}))

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Tool message connection closed (Code: {e.code}, Reason: {e.reason}).")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        logger.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down.")