# Detailed Application Workflow

This document provides a detailed, step-by-step description of the application's workflow, from the initial client connection to the completion of a user's request.

## System Interaction Diagram

The following diagram illustrates the high-level interaction between the system's components:

```
+--------+       +---------------------+       +-----------------+       +----------------+
| Client | ----> | WebSocket Server    | ----> | ClientSession   | ----> | ManagerAgent   |
| (UI)   | <---- | (main.py)           | <---- | (clientSession.py)| <---- | (manager.py)   |
+--------+       +---------------------+       +-----------------+       +----------------+
     ^                     ^                           |                       |
     |                     |                           |                       |
     +---------------------+---------------------------+-----------------------+
                (WebSocket Messages)
```

## Step-by-Step Workflow

### 1. Connection and Session Establishment

The process begins when the client establishes a WebSocket connection with the server.

-   **Client Sends `handshake`:** The client must send an initial message with `type: "handshake"`. To resume a previous session, it can include its `session_id`.
    ```json
    {
      "type": "handshake",
      "session_id": "optional-uuid-here"
    }
    ```
-   **Server Responds:**
    -   If a new session is created, the server responds with a `handshake_response`, a `status` of `created`, and the new `session_id`.
    -   If a session is resumed, the server sends a `handshake` response with the existing `session_id`.
    ```json
    {
      "type": "handshake_response",
      "status": "created",
      "session_id": "new-uuid-goes-here"
    }
    ```
-   **`ClientSession` is Born:** On the server, a dedicated `ClientSession` instance is created to manage this client's state and interaction.

### 2. Sending a Prompt

Once the session is established, the client can send requests to the agent.

-   **Client Sends `prompt`:** The client sends a message with `type: "prompt"`, containing the user's request, context about the document, and any attached files (images, documents).
    ```json
    {
      "type": "prompt",
      "text": "Please add a conclusion to my document.",
      "document_structure": "<html>...</html>",
      "images": [],
      "documents": []
    }
    ```
-   **`ClientSession` Receives:** The `ClientSession` receives this message and calls the `manager.run_prompt()` method, starting the agent's execution cycle.

### 3. Agent Execution and the Interrupt System

This is the core of the agent's operation, where it processes the prompt and interacts with the client via an interrupt-driven system for tool calls.

-   **Agent Runs:** The `ManagerAgent` begins processing the prompt. It can stream back partial text responses, which the `ClientSession` forwards to the client as `msc` (message stream chunk) messages.

-   **Agent Needs a Tool:** When the agent determines it needs to use a tool that requires frontend interaction (like `read_document` or `apply_tool_func`), it doesn't call the tool directly. Instead, it **interrupts** its own execution.

-   **`ClientSession` Catches Interrupt:** The `ClientSession` is designed to catch these interrupts. The interrupt contains a payload with the tool name and its parameters.

-   **`ClientSession` Notifies Client:** The `ClientSession` does two things:
    1.  It sends a `status_update` message to the client, indicating that a tool is about to be executed.
    2.  It sends the interrupt payload to the client as a `tool_call` message.
    ```json
    // Example tool_call message sent to the client
    {
      "type": "tool_call",
      "tool_name": "read_document",
      "as_html": true,
      "timestamp": "...",
      "message": "Reading document",
      "session_id": "..."
    }
    ```
-   **`ClientSession` Waits:** The `ClientSession` now pauses and waits for the client to send back a response.

### 4. Client-Side Tool Execution and Resumption

-   **Client Executes Tool:** The client receives the `tool_call` message and performs the requested action (e.g., reads the document's HTML content).

-   **Client Sends `tool_response`:** After completing the action, the client sends the result back to the server in a `tool_response` message.
    ```json
    {
      "type": "tool_response",
      "content": "<html>...</html>", // The result of the tool call
      "session_id": "..."
    }
    ```

-   **`ClientSession` Resumes Agent:** The `ClientSession` receives the `tool_response` and uses its content to resume the paused `ManagerAgent`. The agent now has the information it needs and continues its execution from where it left off.

-   **Loop:** This interrupt-and-resume cycle can happen multiple times within a single prompt execution.

### 5. Completion

-   **Agent Finishes:** Once the `ManagerAgent` has fully completed its task, the `ClientSession` sends a final `status_update` to the client indicating that the process has ended.
    ```json
    {
        "type": "status_update",
        "process" : "agent",
        "status": "end",
        "message": "Agent processing completed",
        "session_id": "..."
    }
    ```

This workflow creates a robust, decoupled system where the AI agent can operate independently and request services from the client as needed, all while providing a responsive, real-time experience for the user.