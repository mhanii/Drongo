# ContentAgent

The `ContentAgent` is a specialized sub-agent responsible for all content generation tasks within the application. It is called by the `ManagerAgent` whenever a user's request requires the creation of new text or image-related operations. The `ContentAgent` orchestrates its own sub-agents (`HtmlAgent` and `ImageAgent`) to produce high-quality content.

## Role in the Workflow

The `ContentAgent` is not directly exposed to the client. It operates as a service layer called by the `ManagerAgent`.

1.  **Invocation:** The `ManagerAgent` calls the `content_agent.run()` method with a prompt describing the desired content.
2.  **Execution:** The `ContentAgent` uses its internal React agent and a suite of tools to interpret the prompt and generate the content. This may involve calling the `HtmlAgent` to create text and the `ImageAgent` for image operations.
3.  **Content Storage:** As content is generated, it is saved to a database as "content chunks." Each chunk is given a unique `chunk_id`.
4.  **Output:** The `run` method returns a dictionary containing the final status of the operation and a list of the `chunk_id`s that were created. The `ManagerAgent` then uses these IDs to apply the new content to the document.

## Tools

The `ContentAgent` uses the following tools to fulfill content generation requests.

### `run_html_agent(description: str, style_guidelines: str, context: str = "...")`

-   **Purpose:** Generates structured and validated HTML content. This is the primary tool for creating any text-based content, from paragraphs to titles.
-   **Process:**
    -   It takes a detailed `description` of the required content and optional `style_guidelines`.
    -   It invokes the `HtmlAgent`, which handles the core generation logic.
    -   The generated HTML is saved as a `ContentChunk` in the database.
-   **Output:** Returns a string confirming the generation and providing the `chunk_id` of the newly created content.

### `run_image_agent(detailed_instruction: str)`

-   **Purpose:** Manages all image-related tasks. This is a comprehensive tool for everything from adding images to performing complex analysis.
-   **Capabilities:**
    -   **Store Management:** Add, remove, search, and organize images.
    -   **Processing:** Resize, filter, enhance, and convert image formats.
    -   **AI-Powered Analysis:** Generate captions, extract text from images, and analyze visual content.
-   **Usage:** It takes a `detailed_instruction` describing the specific image operation to perform (e.g., "Add an image from URL...", "Generate a caption for image 'abc-123'").
-   **Output:** Returns a string (usually JSON formatted) containing the result of the image operation.

### `analyze_content_requirements(requirements: str)`

-   **Purpose:** Analyzes a user's prompt to determine the best strategy for content generation.
-   **Process:** It uses the underlying AI model to break down a complex request into a structured plan, identifying key elements, image needs, and overall complexity.
-   **Output:** Returns a structured text analysis that guides the agent's next steps.