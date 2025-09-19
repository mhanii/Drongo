import asyncio
import websockets
import json
import uuid
from agents.manager import ManagerAgent
from new_logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
logger = get_logger(DEBUG)

# Global agent store to persist agents across requests
agent_store = {}

async def handler(websocket, path):
    """Main handler for all websocket connections."""
    logger.info(f"New connection established")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "handshake":
                    await handle_handshake(websocket, data)
                elif message_type == "prompt":
                    await handle_prompt(websocket, data)
                elif message_type == "tool_response":
                    await handle_tool_response(websocket, data)
                else:
                    await websocket.send(json.dumps({
                        "error": f"Unknown message type: {message_type}"
                    }))
                    
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=DEBUG)
                await websocket.send(json.dumps({"error": "Server error occurred"}))
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed (Code: {e.code}, Reason: {e.reason})")

async def handle_handshake(websocket, data):
    """Handle handshake requests to establish session."""
    # Check if client provided a session_id to resume
    session_id = data.get("session_id")
    
    if session_id:
        # Validate that the session exists
        if session_id in agent_store:
            logger.info(f"Handshake for session: {session_id}")
            await websocket.send(json.dumps({
                "type": "handshake",
                "session_id": session_id,
            }))
        else:
            logger.info(f"Session not found, creating new session: {session_id}")
            # Create new session with provided ID
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {session_id}")
            await websocket.send(json.dumps({
                "type": "handshake",
                "session_id": session_id,
            }))
    else:
        # Create new session
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        await websocket.send(json.dumps({
            "type": "handshake_response",
            "status": "created",
            "session_id": session_id,
            "message": "New session created"
        }))

async def handle_prompt(websocket, data):
    """Handle prompt requests."""
    # Get or create session
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
    
    # Validate required fields
    required_fields = ["text", "images", "documents", "document_structure"]
    for field in required_fields:
        if field not in data:
            await websocket.send(json.dumps({
                "error": f"Required field '{field}' is missing"
            }))
            return
    
    # Get or create agent for this session
    if session_id not in agent_store:
        message_queue = asyncio.Queue()
        agent = ManagerAgent(
            checkpoint_path="data/manager_checkpoint.sqlite",
            model="models/gemini-2.0-flash",
            store=None,
            queue=message_queue
        )
        agent_store[session_id] = {
            "agent": agent,
            "queue": message_queue
        }
        logger.info(f"Created new agent for session: {session_id}")
    
    agent_data = agent_store[session_id]
    agent = agent_data["agent"]
    message_queue = agent_data["queue"]
    
    # Create sender task for real-time messages
    async def sender():
        while True:
            try:
                message = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                if message is None:  # End signal
                    break
                await websocket.send(message)
                message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in sender: {e}")
                break
    
    sender_task = asyncio.create_task(sender())
    
    try:
        # Run agent
        response = await asyncio.to_thread(agent.run_prompt, data)
        
        # Handle interrupts
        if "__interrupt__" in response:
            interrupt_info = response["__interrupt__"][0]
            logger.info(f"Interrupt info: {interrupt_info}")
        
        # Send final response
        await websocket.send(json.dumps({
            "type": "agent_text",
            "status": "end",
            "content": response.content,
            "session_id": session_id
        }))
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=DEBUG)
        await websocket.send(json.dumps({
            "error": "Agent processing error",
            "session_id": session_id
        }))
    finally:
        # Stop sender
        await message_queue.put(None)
        try:
            await asyncio.wait_for(sender_task, timeout=1.0)
        except asyncio.TimeoutError:
            sender_task.cancel()

async def handle_tool_response(websocket, data):
    """Handle tool response messages."""
    session_id = data.get("session_id")
    if not session_id:
        await websocket.send(json.dumps({"error": "session_id is required"}))
        return
    
    if session_id not in agent_store:
        await websocket.send(json.dumps({
            "error": f"No active session found: {session_id}"
        }))
        return
    
    agent_data = agent_store[session_id]
    agent = agent_data["agent"]
    
    try:
        # Process tool response (implement based on your agent's needs)
        
        logger.info(f"Processed tool response for session {session_id}")
        logger.info(data)
        agent.handle_client_tool_response(data)

        await websocket.send(json.dumps({
            "type": "tool_response_ack",
            "status": "received",
            "session_id": session_id
        }))
        
    except Exception as e:
        logger.error(f"Error handling tool response: {e}", exc_info=DEBUG)
        await websocket.send(json.dumps({
            "error": "Tool response processing error",
            "session_id": session_id
        }))

def cleanup_session(session_id):
    """Clean up a specific session."""
    if session_id in agent_store:
        logger.info(f"Cleaning up session: {session_id}")
        del agent_store[session_id]

async def main():
    """Start the WebSocket server."""
    async with websockets.serve(handler, "localhost", 8765):
        logger.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down")
        # Clean up all sessions
        for session_id in list(agent_store.keys()):
            cleanup_session(session_id)