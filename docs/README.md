# Websocket-based Architecture

The application now uses a websocket-based architecture for real-time, bidirectional communication between the client (frontend) and the server. This has replaced the previous HTTP-based approach, allowing for a more interactive and responsive user experience.

## Endpoints

The websocket server exposes the following endpoints:

*   `ws://<server-address>:<port>/prompt`: This is the primary endpoint for initiating interactions with the AI. The client sends a JSON message to this endpoint containing the user's prompt and any associated data (images, documents, etc.). The server then creates a `ManagerAgent` instance to handle the request.

*   `ws://<server-address>:<port>/tool-message`: This endpoint is used for sending responses from tools back to the `ManagerAgent`. When a tool in the frontend completes a task, it sends a JSON message to this endpoint with the results.

## Frontend Notifications and Tool Calls

A key advantage of using websockets is the ability for the server to proactively notify the frontend of its current state. This is crucial for keeping the user informed about the progress of their request. Here's how it can be achieved:

### Server-to-Frontend Notifications

The server can send messages to the client at any point during the processing of a prompt. This can be used to:

*   **Acknowledge receipt of a prompt:** Immediately after receiving a prompt, the server can send a message to the client to confirm that it has started processing the request.
*   **Provide status updates:** As the `ManagerAgent` progresses through its workflow (e.g., calling different tools, generating content), it can send updates to the frontend. This could include messages like "Generating content...", "Applying changes to the document...", etc.
*   **Signal completion:** Once the entire process is complete, the server can send a final message to the client with the results.

To implement this, the `ManagerAgent` can be modified to accept a websocket connection object. It can then use this object to send messages to the client at various points in its execution.

### Calling Frontend Tools

In some scenarios, it may be necessary for the server to trigger actions in the frontend. For example, the `ManagerAgent` might need to request a piece of information from the user that can only be obtained through a frontend interface (e.g., asking the user to select an area of an image).

This can be achieved by having the server send a specific message to the client that indicates which tool to call and what parameters to use. The frontend would then listen for these messages and execute the appropriate action.

Here's a possible workflow:

1.  The `ManagerAgent` determines that it needs to call a frontend tool.
2.  It sends a JSON message to the client over the websocket connection. This message could look something like this:

    ```json
    {
      "type": "tool_call",
      "tool_name": "image_selection",
      "parameters": {
        "image_id": "some-image-id"
      }
    }
    ```
3.  The frontend receives this message and identifies it as a tool call.
4.  It then calls the `image_selection` tool, passing the specified parameters.
5.  Once the tool has completed its execution (e.g., the user has selected an area of the image), the frontend sends a message back to the server via the `/tool-message` endpoint with the results.

This approach allows for a powerful and flexible integration between the backend AI logic and the frontend user interface.
