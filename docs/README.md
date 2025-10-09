# WebSocket Architecture Overview

This document provides an overview of the WebSocket-based architecture that powers the application, enabling real-time, bidirectional communication between the client and the server.

## Core Components

The architecture is built around three main components:

1.  **WebSocket Server (`main.py`):** The entry point for all client connections. It handles the initial handshake and creates a `ClientSession` for each new connection.

2.  **ClientSession (`session/clientSession.py`):** The central orchestrator for a user's session. Each client is assigned a unique `ClientSession` that manages their state, message queues, and the associated `ManagerAgent`. It acts as a bridge between the WebSocket connection and the agent, ensuring that messages are processed correctly and that the agent's lifecycle is managed.

3.  **ManagerAgent (`agents/manager.py`):** The AI-powered agent responsible for understanding and executing user requests. It uses a suite of tools to generate content, read the document state, and apply changes. The `ManagerAgent` operates independently of the WebSocket connection, communicating with the `ClientSession` through a system of message queues and interrupts.

## Connection and Session Establishment

The communication process begins when a client connects to the WebSocket server. The server and client perform a handshake to establish or resume a session:

1.  **Handshake:** The client sends a `handshake` message, optionally including a `session_id` to resume a previous session.
2.  **Session Creation/Resumption:**
    -   If a valid `session_id` is provided, the server reconnects the WebSocket to the existing `ClientSession`.
    -   Otherwise, a new `session_id` is generated, and a new `ClientSession` is created and stored.
3.  **Handshake Response:** The server sends a `handshake_response` to the client, confirming the session has been created or resumed.

Once the session is established, the client can begin sending `prompt` messages to the server.

## Message Flow and Tool Calls

The `ClientSession` manages the flow of messages between the client and the `ManagerAgent`. This is a simplified, high-level overview of the process:

1.  **Prompt:** The client sends a `prompt` message containing the user's request.
2.  **Processing:** The `ClientSession` receives the prompt and passes it to the `ManagerAgent` for processing.
3.  **Tool Call Interrupt:** When the `ManagerAgent` needs to use a tool that requires client-side action (e.g., reading the document), it triggers an **interrupt**.
4.  **Interrupt Handling:** The `ClientSession` catches the interrupt, sends a `tool_call` message to the client, and waits for a response.
5.  **Tool Response:** The client executes the tool and sends a `tool_response` message back to the server.
6.  **Resuming the Agent:** The `ClientSession` receives the response and uses it to resume the `ManagerAgent`, which then continues its work.
7.  **Streaming Updates:** Throughout this process, the `ClientSession` streams status updates and partial message responses from the agent to the client, providing real-time feedback.

This architecture decouples the agent's logic from the WebSocket communication, allowing for a more robust and scalable system. For a more detailed breakdown of the entire workflow, please see the [Workflow](./workflow.md) documentation.