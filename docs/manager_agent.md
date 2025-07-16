# Manager Agent

The `ManagerAgent` is the central orchestrator of the document editing application. It is responsible for translating user requests into a sequence of tool calls that modify the document. It does not directly edit the document content itself, but rather delegates content creation and modification to other agents and tools.

## Inputs

The `ManagerAgent` takes a dictionary as input, which can contain the following keys:

- `text`: The user's prompt, describing the desired change.
- `images`: A list of images to be used as context. Each image is a dictionary with `content` (base64 encoded) and `name` keys.
- `documents`: A list of documents to be used as context. Each document is a dictionary with `content` (base64 encoded) and `name` keys.
- `document_structure`: An HTML string representing the current state of the document.

## Workflow

1.  **Analyze:** The agent first analyzes the user's request to determine the intent: adding, deleting, or editing content.
2.  **Generate (optional):** If the request requires new content, the `ManagerAgent` calls the `generate_content` tool. This tool delegates the content creation to a specialized writing agent and returns a list of content "chunks" with unique IDs.
3.  **Apply:** The `ManagerAgent` then calls the `apply_tool_func` tool to modify the document. This tool takes the action type (`INSERT`, `DELETE`, or `EDIT`), the target location (a CSS selector), and, for `INSERT` and `EDIT` actions, the `chunk_id` of the content to be used.

## Tools

### `generate_content(description: str, style_guidelines: str)`

-   **Purpose:** Delegates content creation to a specialized writing agent.
-   **Use When:** The user asks to write, create, generate, summarize, or add any new text.
-   **Output:** Returns a list of one or more generated content chunks, each with a unique `chunk_id`.

### `apply_tool_func(action_type: str, target_location: str, chunk_id: Optional[str] = None, relative_position: Optional[str] = None)`

-   **Purpose:** Applies a structural change to the document.
-   **Parameters:**
    -   `action_type` (str): **REQUIRED.** Must be one of `"INSERT"`, `"DELETE"`, or `"EDIT"`.
    -   `target_location` (str): **REQUIRED.** A precise CSS selector or description identifying the target HTML element.
    -   `chunk_id` (str): **REQUIRED for 'INSERT' and 'EDIT'**. The `chunk_id` of the content received from the `generate_content` tool.
    -   `relative_position` (str): **REQUIRED *only* when `action_type` is "INSERT"**. Must be one of `"BEFORE"` or `"AFTER"`.

## Outputs

The `ManagerAgent`'s `run_prompt` method returns the final message from the agent's execution, which is typically the result of the `apply_tool_func` call. This result is a dictionary indicating the status of the operation (e.g., `{"status": "success", ...}`).
