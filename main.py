import asyncio
import websockets
import json
import uuid
from agents.manager import ManagerAgent
from new_logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()

DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
logger = get_logger(DEBUG)


agent_store = {}

async def handler(websocket, path):
    """Main handler for all websocket connections."""
    session_id = None 
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                session_id = data.get("session_id",None) 
                
                message_type = data.get("type")
                
                if message_type == "handshake":
                    session_id = await handle_handshake(websocket, data)
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
                logger.error(f"Error processing message: {e}")
                await websocket.send(json.dumps({"error": "Server error occurred"}))
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed (Code: {e.code}, Reason: {e.reason})")
    finally:
        if session_id:
            await cleanup_session(session_id)

async def handle_handshake(websocket, data):
    """Handle handshake requests to establish session."""
    # Check if client provided a session_id to resume
    session_id = data.get("session_id") or str(uuid.uuid4())
    

    if session_id not in agent_store:
        # Create new session
        logger.info(f"Created new session: {session_id}")

        message_queue = asyncio.Queue()
        sender_task = asyncio.create_task(session_sender(websocket, message_queue))

        agent_store[session_id] = {
            "agent": None,
            "queue": message_queue,
            "sender_task": sender_task
        }
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
    if not session_id or session_id not in agent_store:
        logger.error(f"Session is not initialized correctly")
        return
    
    logger.info(f"Recieved prompt {data['text']} from session id {session_id}")
    required_fields = ["text", "images", "documents", "document_structure"]
    for field in required_fields:
        if field not in data:
            await websocket.send(json.dumps({
                "error": f"Required field '{field}' is missing"
            }))
            return
    
    # Get or create agent for this session
    if agent_store[session_id].get("agent") == None:
        agent = ManagerAgent(
            checkpoint_path="data/manager_checkpoint.sqlite",
            model="models/gemini-2.0-flash",
            store=None,
            queue=agent_store[session_id].get("queue")
        )
        agent_store[session_id]["agent"] = agent
        logger.info(f"Created new agent for session: {session_id}")
    
    agent_data = agent_store[session_id]
    agent = agent_data["agent"]
    message_queue = agent_data["queue"]
    
    
    try:
        # Run agent
        response = await asyncio.to_thread(agent.run_prompt, data)

        await websocket.send(json.dumps({
            "type": "agent_text",
            "status": "end",
            "content": response.content,
            "session_id": session_id
        }))
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=DEBUG)
        await websocket.send(json.dumps({
            "error": str(e),
            "session_id": session_id
        }))


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
        await websocket.send(json.dumps({
            "type": "tool_response_ack",
            "status": "received",
            "session_id": session_id
        }))
        await asyncio.to_thread(agent.handle_client_tool_response,data)



        
    except Exception as e:
        logger.error(f"Error handling tool response: {e}", exc_info=DEBUG)
        await websocket.send(json.dumps({
            "error": "Tool response processing error",
            "session_id": session_id
        }))

async def session_sender(websocket, queue):
    """A dedicated, long-lived sender for a single WebSocket session."""
    while True:
        try:
            message = await queue.get()
            if message is None:  # End signal
                logger.info("Sender task received None signal, shutting down.")
                break
            await websocket.send(message)
            queue.task_done()
        except asyncio.CancelledError:
            logger.info("Sender task was cancelled.")
            break
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Sender task exiting because connection is closed.")
            break
        except Exception as e:
            logger.error(f"Error in session_sender: {e}")
            break

async def cleanup_session(session_id):
    """Clean up a specific session by stopping its tasks."""
    if session_id in agent_store:
        logger.info(f"Cleaning up session: {session_id}")
        session_data = agent_store[session_id]
        
        # Gracefully stop the sender task
        sender_task = session_data["sender_task"]
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass # Expected

        del agent_store[session_id]
        logger.info(f"Session {session_id} cleaned up.")

async def main():
    """Start the WebSocket server."""
    try:
        async with websockets.serve(handler, "localhost", 8765):
            logger.info("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # run forever

    except KeyboardInterrupt:
        logger.info("Server shutting down")
        # Clean up all sessions
        async for session_id in list(agent_store.keys()):
            await cleanup_session(session_id)
if __name__ == "__main__":
    asyncio.run(main())
