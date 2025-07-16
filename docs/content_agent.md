# Content Agent

The `ContentAgent` is a specialized agent responsible for generating and managing content, including HTML and images. It acts as a central hub for all content-related operations, coordinating between specialized sub-agents for HTML and image generation.

## Inputs

The `ContentAgent`'s `run` method takes the following arguments:

- `prompt`: A string describing the content to be generated.
- `document_structure`: An HTML string representing the current state of the document.

## Workflow

1.  **Analyze Request:** The agent analyzes the user's prompt to determine the type of content required (HTML, images, or both).
2.  **Delegate to Sub-Agents:** Based on the analysis, the `ContentAgent` delegates the task to the appropriate sub-agent:
    -   `HtmlAgent`: For generating structured HTML content.
    -   `ImageAgent`: For all image-related operations.
3.  **Content Integration:** The agent ensures that the generated text and visual content are seamlessly integrated.
4.  **Quality Assurance:** The agent ensures that all generated content meets the specified requirements and quality standards.

## Tools

### `run_html_agent(description: str, style_guidelines: str, context: str = "No additional context") -> str`

-   **Purpose:** Runs the `HtmlAgent` to generate high-quality HTML content.
-   **Output:** Returns a string containing the generated HTML, a `chunk_id`, and the status of the operation.

### `run_image_agent(detailed_instruction: str) -> str`

-   **Purpose:** Runs the `ImageAgent` for comprehensive image operations and management.
-   **Output:** Returns a string containing the result of the image agent operation, usually in JSON format.

### `analyze_content_requirements(requirements: str) -> str`

-   **Purpose:** Analyzes content requirements to determine the best approach for generation.
-   **Output:** Returns a structured analysis of the content requirements.

## Outputs

The `ContentAgent`'s `run` method returns a list of "content chunks." Each chunk is a dictionary containing the generated content, its ID, and its status. If no content is generated, or if all generation attempts result in errors, an appropriate message is returned.
