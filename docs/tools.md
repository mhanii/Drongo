# Tools

This document describes the tools used by the agents in the document editing application.

## ManagerAgent Tools

### `generate_content(description: str, style_guidelines: str)`

-   **Purpose:** Delegates content creation to a specialized writing agent.
-   **Use When:** The user asks to write, create, generate, summarize, or add any new text.
-   **Output:** Returns a list of one or more generated content chunks, each with a unique `chunk_id`.

### `apply_tool_func(action_type: str, chunk_id: Optional[str] = None)`

-   **Purpose:** Applies a structural change to the document.
-   **Parameters:**
    -   `action_type` (str): **REQUIRED.** Must be one of `"INSERT"`, `"DELETE"`, or `"EDIT"`.
    -   `chunk_id` (str): **REQUIRED for 'INSERT' and 'EDIT'**. The `chunk_id` of the content received from the `generate_content` tool.

## ContentAgent Tools

### `run_html_agent(description: str, style_guidelines: str, context: str = "No additional context") -> str`

-   **Purpose:** Runs the `HtmlAgent` to generate high-quality HTML content.
-   **Output:** Returns a string containing the generated HTML, a `chunk_id`, and the status of the operation.

### `run_image_agent(detailed_instruction: str) -> str`

-   **Purpose:** Runs the `ImageAgent` for comprehensive image operations and management.
-   **Output:** Returns a string containing the result of the image agent operation, usually in JSON format.

### `analyze_content_requirements(requirements: str) -> str`

-   **Purpose:** Analyzes content requirements to determine the best approach for generation.
-   **Output:** Returns a structured analysis of the content requirements.
