from agents.manager import ManagerAgent
from new_logger import get_logger
import asyncio
import json
import uuid
import websockets

logger = get_logger()

class ClientSession:
    """
    An actor that manages the state and logic for a single websocket client.
    """
    def __init__(self, websocket, session_id):
        self.websocket = websocket
        self.session_id = session_id
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.resume_future = None
        
        
        # Initialize the agent for this session
        self.manager = ManagerAgent(
            checkpoint_path=f"data/{session_id}_checkpoint.sqlite", # Session-specific checkpoint
            model="models/gemini-2.5-flash",
            store=None, # Should be initialized properly if needed
            queue=asyncio.Queue() # Agent's own queue for streaming messages
        )
        # The agent's message queue for streaming partial responses
        self.agent_message_queue = self.manager.queue
        self.sender_task = None

        logger.info(f"ClientSession created for session: {self.session_id}")

    def start(self):
        """Starts the background tasks for this session."""
        self.processing_task = asyncio.create_task(self._message_processor())
        self.sender_task = asyncio.create_task(self._sender())
        logger.info(f"Started processing and sender tasks for session: {self.session_id}")

    async def stop(self):
        """Gracefully stops the session's background tasks."""
        if self.processing_task:
            self.processing_task.cancel()
        if self.sender_task:
            # Signal the sender to stop
            await self.agent_message_queue.put(None)
            self.sender_task.cancel()
        logger.info(f"Stopped tasks for session: {self.session_id}")



    async def receive_message(self, data: dict):
        message_type = data.get("type")
        # Those two are seperate for a reason - tool_response needs to directly resolve the future
        # Don't try to merge receive_message and _message_processor because a block in _message_processor would cause deadlocks. I am learning a lot by myself lol.
        if message_type == "tool_response":
            self.resolve_interrupt(data)
        else:
            await self._enqueue_message(data)

    def resolve_interrupt(self, data: dict):
        """Directly resolves the pending future."""
        if self.resume_future and not self.resume_future.done():
            self.resume_future.set_result(data)
            logger.info(f"Directly resolved interrupt for session {self.session_id}.")
            return True
        logger.warning(f"Received interrupt resolution for {self.session_id}, but nothing was pending.")
        return False

    async def _enqueue_message(self, message):
        """Puts a message from the client into the processing queue."""
        await self.message_queue.put(message)

    async def _sender(self):
        """Forwards messages from the agent's queue to the websocket client."""
        while True:
            try:
                message = await self.agent_message_queue.get()
                if message is None: # End signal
                    break
                await self.websocket.send(message)
                self.agent_message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Sender {self.session_id}] Error: {e}")
                break

    async def _message_processor(self):
        """The main loop that processes incoming messages from the mailbox."""
        while True:
            try:
                data = await self.message_queue.get()
                message_type = data.get("type")
                
                logger.info(f"Processing message of type '{message_type}' for session {self.session_id}")

                if message_type == "prompt":
                    await self._handle_prompt(data)
                elif message_type == "tool_response":
                    await self._handle_tool_response(data)
                # Other message types can be handled here
                else:
                    await self.websocket.send(json.dumps({
                        "error": f"Unknown message type: {message_type}",
                        "session_id": self.session_id
                    }))
            
            except asyncio.CancelledError:
                logger.info(f"Message processor for {self.session_id} cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in message processor for {self.session_id}: {e}", exc_info=DEBUG)
                await self.websocket.send(json.dumps({
                    "error": "Server processing error",
                    "session_id": self.session_id
                }))

    async def _handle_prompt(self, data):
        """Handles a new prompt from the client with a restartable generator loop."""
        logger.info(f"Handling prompt for session {self.session_id}")

        # Start with the initial generator
        generator = self.manager.run_prompt(data)

        while True: # Loop to handle generator switches
            try:
                async for chunk in generator:
                    # logger.debug(f"Sending chunk to client for session {self.session_id}: {chunk}")

                    if chunk['type'] == "interrupt":
                        logger.info(f"Session {self.session_id} interrupted, waiting for tool response.")

                        
                        payload = dict(chunk["payload"])  # Make a copy to avoid mutating the original
                        payload["session_id"] = self.session_id
                        await self.websocket.send(json.dumps({
                            "type": "status_update",
                            "process" : payload.get("tool_name", "unknown"),
                            "status": "begin",
                            "message": payload.get("message", "Tool execution started"),
                            "session_id": self.session_id
                        }))
                        await self.websocket.send(json.dumps(payload))

                        self.resume_future = asyncio.get_event_loop().create_future()
                        tool_response_data = await self.resume_future
                        logger.info(f"Resuming session {self.session_id} with tool response: {tool_response_data}")
                        self.resume_future = None  # Clear the future after use

                        await self.websocket.send(json.dumps({
                            "type": "status_update",
                            "process" : payload.get("tool_name", "unknown"),
                            "status": "end",
                            "message": payload.get("message", "Tool execution ended"),
                            "session_id": self.session_id
                        }))
                        # Get the NEW generator for the resumed agent
                        generator = self.manager.resume_agent(tool_response_data)
                        break 

                    elif chunk["type"] == "message":
                        payload = dict(chunk["payload"])  # Make a copy to avoid mutating the original
                        payload["type"] = "msc" # message stream chunk
                        payload["session_id"] = self.session_id
                        await self.websocket.send(json.dumps(payload))

                else: 
                    logger.info(f"Generator for session {self.session_id} finished normally.")
                    await self.websocket.send(json.dumps({
                        "type": "status_update",
                        "process" : "agent",
                        "status": "end",
                        "message": "Agent processing completed",
                        "session_id": self.session_id
                    }))

                    break # Exit the outer 'while' loop

            except Exception as e:
                logger.error(f"Error running agent for session {self.session_id}: {e}", exc_info=True)
                await self.websocket.send(json.dumps({
                    "error": "Agent processing error",
                    "session_id": self.session_id
                }))
                break # Exit the loop on errorr after tool response.")






            except Exception as e:
                logger.error(f"Error running agent for session {self.session_id}: {e}", exc_info=DEBUG)
                await self.websocket.send(json.dumps({
                    "type":"status_update",
                    "process" : "agent",
                    "status": "error",
                    "error": "Agent processing error",
                    "session_id": self.session_id
                }))

    async def _handle_tool_response(self, data):
        """Handles a tool response from the client to resume a paused agent."""
        if self.resume_future and not self.resume_future.done():
            self.resume_future.set_result(data)
            logger.info(f"Resumed session {self.session_id} with tool response.")
            # Acknowledge receipt
            await self.websocket.send(json.dumps({
                "type": "tool_response_ack", "status": "received", "session_id": self.session_id
            }))
        else:
            logger.warning(f"Received tool_response for {self.session_id} but no operation was pending.")



    def cleanup_session(self):
        """Clean up a specific session."""
        logger.info(f"Cleaning up session: {self.session_id}")
        
        if self.sender_task:
            if self.message_queue:
                self.message_queue.put_nowait(None)  # Signal sender to stop
            self.sender_task.cancel()
        if self.processing_task:
            self.processing_task.cancel()   
        # Additional cleanup if needed