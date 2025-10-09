# ManagerAgent

The `ManagerAgent` is the AI-powered core of the application, acting as the central orchestrator for fulfilling user requests. It is responsible for understanding a user's intent and translating it into a series of tool calls to read, generate, and modify the document.

The `ManagerAgent` is instantiated within a `ClientSession` and does not interact directly with the WebSocket. Instead, it communicates through a system of yielded messages and interrupts, as described in the [Workflow](./workflow.md) documentation.

## Execution Model: Streaming and Interrupts

The `ManagerAgent` runs asynchronously and streams its output. The `run_prompt` method returns a generator that yields different types of data:

-   **Message Chunks:** Partial text responses from the agent, which are streamed to the client for a real-time conversational experience.
-   **Interrupts:** When the agent needs to use a tool, it halts its own execution and yields an `interrupt` payload. The `ClientSession` catches this and sends a corresponding `tool_call` to the client. The agent remains paused until the `ClientSession` receives a `tool_response` and resumes it.

## Tools

The `ManagerAgent` has access to a suite of tools to perform its tasks. It decides which tool to use based on the user's prompt and the current context.

### `read_document(asHTML: bool = False)`

-   **Purpose:** To retrieve the current state of the document. This is crucial for the agent to understand the context before making changes.
-   **Parameters:**
    -   `asHTML` (bool): If `True`, returns the document as an HTML string. This is useful for determining the precise positions of elements for modification. If `False` (default), returns the document as plain text.
-   **Interrupt Payload:** When this tool is called, it triggers an interrupt with the `tool_name: "read_document"`.

### `generate_content(description: str, style_guidelines: str = "")`

-   **Purpose:** Delegates the task of creating new content to the specialized `ContentAgent`. The `ManagerAgent` should use this tool whenever the user asks to write, create, add, or generate new text. It should not generate content by itself.
-   **Parameters:**
    -   `description` (str): A detailed description of the content that needs to be generated.
    -   `style_guidelines` (str, optional): Any specific style instructions to follow.
-   **Output:** This tool does not directly return a value to the agent. It runs the `ContentAgent`, which generates content and stores it in a database. The most important output is the `chunk_id`, which is returned to the agent and used to refer to the newly created content in subsequent tool calls.

### `apply_tool_func(action_type: str, chunk_id: str = "", hint: str = "")`

-   **Purpose:** Applies a change to the document, such as inserting, deleting, or editing content. This tool first uses the `FindLocationTool` to determine *where* the change should be applied.
-   **Parameters:**
    -   `action_type` (str): **REQUIRED.** The type of action to perform. Must be one of `INSERT`, `DELETE`, or `EDIT`.
    -   `chunk_id` (str): **REQUIRED for `INSERT` and `EDIT` actions.** The ID of the content chunk (obtained from `generate_content`) to be inserted or used for editing.
    -   `hint` (str, optional): Additional information to help the `FindLocationTool` determine the correct position for the change.
-   **Interrupt Payload:** Triggers an interrupt with `tool_name: "apply"`. The payload includes the `action_type`, the resolved start and end positions, and the HTML of the chunk to be applied.

### `fast_apply(action_type: str, start_position: int, end_position: int, text: str = "")`

-   **Purpose:** A more direct version of `apply_tool_func` for making small, explicit text changes (e.g., fixing a typo, adding a single word). It bypasses the `FindLocationTool` and applies the change directly at the specified positions.
-   **Use Case:** This tool should only be used when the agent knows the *exact* start and end positions for the change, typically after calling `read_document` with `asHTML=True`.
-   **Parameters:**
    -   `action_type` (str): The action type: `INSERT`, `DELETE`, or `EDIT`.
    -   `start_position` (int): The starting character index for the change.
    -   `end_position` (int): The ending character index for the change.
    -   `text` (str): The text to insert or use for editing.
-   **Interrupt Payload:** Triggers an interrupt with `tool_name: "apply"`, similar to `apply_tool_func`.