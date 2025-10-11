# AI-Powered Document Editing Assistant

This project is a sophisticated, AI-powered document editing assistant that operates over a real-time WebSocket connection. It uses a multi-agent architecture to understand user requests, generate content, and modify documents in a collaborative environment.

## Project Purpose

The primary goal of this application is to provide a seamless and intelligent document editing experience. Users can interact with an AI assistant in natural language to perform a wide range of tasks, from writing and formatting text to managing images and applying complex structural changes to a document. The system is designed to be highly responsive and interactive, providing real-time feedback and updates to the user.

## Core Architecture

The application is built on a robust, session-based WebSocket architecture that decouples the AI logic from the communication layer.

-   **WebSocket Server (`main.py`):** The entry point for all client connections. It manages the lifecycle of user sessions.
-   **ClientSession (`session/clientSession.py`):** The heart of the session management. Each connected user is assigned a `ClientSession`, which acts as an orchestrator, managing the state, message flow, and the AI agents for that user.
-   **Multi-Agent System (`agents/`):** The "brain" of the application. It consists of a hierarchy of agents:
    -   **`ManagerAgent`:** The primary agent that interprets the user's intent and coordinates the overall task.
    -   **`ContentAgent`:** A specialized sub-agent responsible for all content generation, including text and images.

## Workflow Overview

The system uses an interrupt-driven workflow to communicate between the AI agents and the client, enabling the agents to request information or actions from the user's side.

1.  **Connection:** A client connects to the WebSocket server and establishes a session.
2.  **Prompt:** The user sends a prompt (e.g., "Add a title to my document").
3.  **Agent Execution:** The `ManagerAgent` begins processing the request.
4.  **Interrupt for Tools:** If the agent needs to perform an action that requires client-side information (like reading the current document content or applying a change), it triggers an **interrupt**.
5.  **Client-Side Execution:** The client receives the interrupt as a `tool_call`, executes the required action, and sends the result back in a `tool_response`.
6.  **Resumption:** The agent resumes its work with the new information.
7.  **Completion:** The process continues until the user's request is fully addressed, with status updates streamed to the client throughout.

## Documentation

For a deeper understanding of the project's architecture, components, and workflow, please refer to the detailed documentation in the `docs/` directory:

-   **[WebSocket Architecture](./docs/README.md):** An overview of the session-based WebSocket setup.
-   **[Detailed Workflow](./docs/workflow.md):** A step-by-step guide to the communication flow and the interrupt system.
-   **[ManagerAgent](./docs/manager_agent.md):** Documentation for the primary orchestrating agent and its tools.
-   **[ContentAgent](./docs/content_agent.md):** Documentation for the specialized content generation agent.