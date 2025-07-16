# Workflow

The document editing application uses a multi-agent system to handle user requests. The workflow is orchestrated by the `ManagerAgent`, which delegates tasks to specialized agents like the `ContentAgent`.

## High-Level Workflow

1.  **User Request:** The user provides a request to the application, which can include text, images, and documents.
2.  **Request Handling:** The `ManagerAgent` receives the request and its context (the current document structure).
3.  **Analysis and Delegation:** The `ManagerAgent` analyzes the user's prompt to understand their intent.
    -   If the request requires generating new content, the `ManagerAgent` calls the `ContentAgent`.
    -   The `ContentAgent` then uses its own sub-agents (`HtmlAgent` and `ImageAgent`) to generate the required content.
    -   The `ContentAgent` returns the generated content to the `ManagerAgent` in the form of "content chunks."
4.  **Document Modification:** The `ManagerAgent` uses the `apply_tool_func` tool to insert, delete, or edit content in the document. This tool takes the content chunks from the `ContentAgent` and applies them to the document structure.
5.  **Response:** The `ManagerAgent` returns a status message indicating the result of the operation.

## Agent Interaction Diagram

```
[User] -> [ManagerAgent]
[ManagerAgent] -> [ContentAgent]
[ContentAgent] -> [HtmlAgent]
[ContentAgent] -> [ImageAgent]
[HtmlAgent] -> [ContentAgent]
[ImageAgent] -> [ContentAgent]
[ContentAgent] -> [ManagerAgent]
[ManagerAgent] -> [Document]
```
