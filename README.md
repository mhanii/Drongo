# Decoupling WebSockets from Synchronous Code with Message Queues

This document outlines the process of decoupling WebSocket communication from synchronous business logic using a message queue approach. This pattern enhances modularity, maintainability, and allows for non-blocking communication between synchronous and asynchronous components.

## The Message Queue Approach

The core idea is to use a message queue as an intermediary between synchronous functions and the asynchronous WebSocket handler. This acts as a translator and messenger, allowing them to communicate without forcing the synchronous code to become asynchronous.

### How It Works

1.  **Client Connection and Queue Creation**: When a client connects via WebSocket, the async handler creates a dedicated `asyncio.Queue` for that client.

2.  **Passing the Queue**: Instead of passing the WebSocket object to business logic, the queue is passed. The queue can be safely accessed from both synchronous and asynchronous code.

3.  **Synchronous Logic Puts Messages**: Synchronous functions can put update messages into the queue as they execute their tasks. This is a non-blocking operation.

4.  **Asynchronous Handler Sends Messages**: The async WebSocket handler continuously monitors the queue for new messages. When a message appears, it's retrieved and sent to the client over the WebSocket.

## Implementation Details

### `asyncio.to_thread`

To prevent the synchronous agent logic from blocking the main asyncio event loop, we use `asyncio.to_thread`. This function runs the synchronous function in a separate thread, ensuring the WebSocket server remains responsive.

For example, the agent's `run_prompt` method is called like this:
```python
response = await asyncio.to_thread(agent.run_prompt, input_data)
```

This was a crucial change to keep the server running smoothly while handling potentially long-running synchronous tasks.
