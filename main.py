import asyncio
import websockets
import json
from agents.manager import ManagerAgent
from new_logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
logger = get_logger(DEBUG)

agent_store = {}

async def handler(websocket, path):
    logger.info(f"New connection on path: {path}")
    if path == "/prompt":
        await handle_prompt(websocket)
    elif path == "/tool-message":
        await handle_tool_message(websocket)
    else:
        logger.warning(f"Unknown path: {path}")
        await websocket.close()

async def handle_prompt(websocket):
    session_id = str(id(websocket))  # Simple session management
    agent = agent_store.get(session_id)
    if agent is None:
        agent = ManagerAgent(
            checkpoint_path="data/manager_checkpoint.sqlite",
            model="models/gemini-2.5-pro",
            store=None
        )
        agent_store[session_id] = agent

    try:
        async for message in websocket:
            try:
                input_data = json.loads(message)
                # Validate input_data structure
                if "text" not in input_data:
                    raise ValueError("'text' field is missing in the input data")

                response = agent.run_prompt(input_data)
                logger.info(response)
                await websocket.send(json.dumps({
                    "message": "Prompt processed successfully.",
                    "agent_response": response.content
                }))
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message.")
                await websocket.send(json.dumps({"error": "Invalid JSON format."}))
            except ValueError as ve:
                logger.error(f"Invalid input data: {ve}")
                await websocket.send(json.dumps({"error": str(ve)}))
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                await websocket.send(json.dumps({"error": "An unexpected error occurred."}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed for session {session_id}")
    finally:
        if session_id in agent_store:
            del agent_store[session_id]

async def handle_tool_message(websocket):
    try:
        async for message in websocket:
            try:
                tool_data = json.loads(message)
                session_id = tool_data.get("session_id")
                agent = agent_store.get(session_id)

                if not agent:
                    await websocket.send(json.dumps({"error": "Agent not found for this session."}))
                    continue

                # Assuming agent has a method to handle tool responses
                # This part is a placeholder for the actual logic
                # agent.handle_tool_response(tool_data)

                logger.info(f"Received tool message for session {session_id}: {tool_data}")
                await websocket.send(json.dumps({"status": "received"}))

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message.")
                await websocket.send(json.dumps({"error": "Invalid JSON format."}))
            except Exception as e:
                logger.error(f"An error occurred in handle_tool_message: {e}")
                await websocket.send(json.dumps({"error": "An unexpected error occurred."}))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Tool message connection closed.")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        logger.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
