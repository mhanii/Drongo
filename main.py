import asyncio
import websockets
import json
import uuid
from agents.manager import ManagerAgent
from new_logger import get_logger
from dotenv import load_dotenv
from session.clientSession import ClientSession
import os

load_dotenv()

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
logger = get_logger(DEBUG)

# Global agent store to persist agents across requests
session_store = {}

async def handler(websocket, path):
    """Main handler, acts as a message router to the correct ClientSession."""
    session_id = None
    session = None
    logger.info("New connection established.")
    try:
        # The first message must be a handshake to establish or resume a session
        handshake_message = await websocket.recv()
        logger.debug(f"Received handshake message: {handshake_message}")
        data = json.loads(handshake_message)

        if data.get("type") != "handshake":
            logger.warning("Handshake not received, closing connection.")
            await websocket.send(json.dumps({"error": "Handshake required"}))
            return

        session_id = data.get("session_id")
        if session_id and session_id in session_store:
            session = session_store[session_id]
            session.websocket = websocket # Re-attach to new websocket connection
            logger.info(f"Resumed session: {session_id}")
            await websocket.send(json.dumps({"type": "handshake", "session_id": session_id}))
        else:
            new_session_id = str(uuid.uuid4())
            session = ClientSession(websocket, new_session_id)
            session_store[new_session_id] = session
            session.start()
            logger.info(f"Created new session: {new_session_id}")
            await websocket.send(json.dumps({
                "type": "handshake_response", "status": "created", "session_id": new_session_id
            }))

        # Main message routing loop
        async for message in websocket:
            logger.debug(f"Received message: {message}")
            try:
                message_data = json.loads(message)
                # Route the message to the session's personal mailbox
                await session.receive_message(message_data)
                logger.info(f"Message enqueued for session {session.session_id}")
            except json.JSONDecodeError:
                logger.error("Invalid JSON format received.")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed for session {session_id} (Code: {e.code})")
    except Exception as e:
        logger.error(f"Unexpected error in handler: {e}", exc_info=DEBUG)
    finally:
        # On disconnect, we don't necessarily want to delete the session,
        # as the user might reconnect. We can implement a timeout cleanup policy.
        # For now, we'll just log it.
        if session:
            logger.info(f"Client for session {session.session_id} disconnected.")


async def handle_prompt(websocket, data):
    """Handle prompt requests."""
    logger.info("Handling prompt request.")
    # Get or create session
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
    # Validate required fields
    required_fields = ["text", "images", "documents", "document_structure"]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Required field '{field}' is missing in prompt.")
            await websocket.send(json.dumps({
                "error": f"Required field '{field}' is missing"
            }))
            return
    # Get or create agent for this session
    if session_id not in session_store:
        message_queue = asyncio.Queue()
        agent = ManagerAgent(
            checkpoint_path="data/manager_checkpoint.sqlite",
            model="models/gemini-2.0-flash",
            store=None,
            queue=message_queue
        )
        session_store[session_id] = {
            "agent": agent,
            "queue": message_queue
        }
        logger.info(f"Created new agent for session: {session_id}")
    agent_data = session_store[session_id]
    agent = agent_data["agent"]
    message_queue = agent_data["queue"]
    sender_task = asyncio.create_task(sender(websocket,message_queue))
    try:
        logger.debug(f"Running agent prompt for session {session_id}")
        response = await asyncio.to_thread(agent.run_prompt, data)
        while response.get("__interrupt__"):
            interrupt_payload = response["__interrupt__"]
            logger.info(f"Handling interrupt for session {session_id}: {interrupt_payload}")
            serializable_payload = interrupt_payload[0].value
            await websocket.send(json.dumps({
                "type": "interrupt",
                "payload": serializable_payload,
                "session_id": session_id
            }))
            agent_data["resume_future"] = asyncio.get_event_loop().create_future()
            tool_response = await agent_data["resume_future"]
            response = await asyncio.to_thread(agent.handle_client_tool_response, tool_response)
        logger.info(f"Sending final agent response for session {session_id}")
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

async def sender(websocket,message_queue):
    logger.info("Sender task started.")
    while True:
        try:
            message = await asyncio.wait_for(message_queue.get(), timeout=0.1)
            if message is None:  # End signal
                logger.info("Sender received end signal.")
                break
            logger.debug(f"Sender sending message: {message}")
            await websocket.send(message)
            message_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Error in sender: {e}")
            break

async def handle_tool_response(websocket, data):
    """Handle tool response messages."""
    logger.info("Handling tool response.")
    session_id = data.get("session_id")
    if not session_id:
        logger.warning("No session_id provided in tool response.")
        await websocket.send(json.dumps({"error": "session_id is required"}))
        return
    if session_id not in session_store:
        logger.warning(f"No active session found for tool response: {session_id}")
        await websocket.send(json.dumps({
            "error": f"No active session found: {session_id}"
        }))
        return
    agent_data = session_store[session_id]
    agent = agent_data["agent"]
    try:
        resume_future = agent_data.get("resume_future")
        if resume_future and not resume_future.done():
            logger.info(f"Resuming session {session_id} with tool response {data}")
            resume_future.set_result(data)
        else:
            logger.info(f"No pending interrupt for session {session_id}")
        logger.info(f"Processed tool response for session {session_id}")
        logger.debug(f"Tool response data: {data}")
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
# Create sender task for real-time messages
async def sender(websocket, message_queue):
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

def cleanup_session(session_id):
    logger.info(f"Cleaning up session: {session_id}")
    session = session_store.get(session_id)
    if not session:
        logger.warning(f"No session found to clean up: {session_id}")
        return
    session.cleanup_session()
    del session_store[session_id]


async def main():
    """Start the WebSocket server."""
    logger.info("Starting WebSocket server...")
    async with websockets.serve(handler, "localhost", 8765):
        logger.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        logger.info("Server is starting.")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down")
        # Clean up all sessions
        for session_id in list(session_store.keys()):
            cleanup_session(session_id)