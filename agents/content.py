from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from context.store import ContextStore
from typing_extensions import TypedDict
from typing import Dict
import sqlite3
from typing import Annotated, List
from agents.sub_agents.image import ImageAgent
from agents.sub_agents.html import HtmlAgent
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from database.content_chunk_db import ContentChunk, ContentChunkDB
from logger import log_content_agent_event

class State(TypedDict):
    messages: Annotated[List, add_messages]

# Removed ContentChunk class from here

class ContentAgent:
    def __init__(self, model: str, checkpoint_path: str) -> None:
        self.connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.chunk_db = ContentChunkDB(self.connection)
        # self.CS = store
        self.model_instance = ChatGoogleGenerativeAI(model=model)
        self.html_agent = HtmlAgent(model=self.model_instance, checkpoint_path=checkpoint_path)
        self.image_agent = ImageAgent(model=self.model_instance, checkpoint_path=checkpoint_path)
        self.document_structure = ""
        self.generated_chunks: List[Dict] = []
        self.config = {"configurable": {"thread_id": "1"}}

        
        self.defined_tools = [
            self.run_html_agent,
            self.run_image_agent,
            # self.get_context_summary,
            # self.coordinate_content_generation,
            # self.analyze_content_requirements
        ]

        self.agent = create_react_agent(
            model=self.model_instance,
            tools=self.defined_tools,
            debug=False,
            # checkpointer=SqliteSaver(self.connection),
            checkpointer=MemorySaver(),

            prompt="""You are an advanced Content Generation Agent and coordinator. You are the central hub for all content-related operations in this system. Your primary responsibilities include:

## CORE RESPONSIBILITIES:
1. **Content Strategy & Planning**: Analyze user requirements and determine the best approach for content generation
2. **HTML Content Generation**: Create structured, high-quality HTML content using the HTML agent
3. **Image Management & Processing**: Handle all image-related operations through the advanced Image agent
4. **Content Integration**: Seamlessly integrate text and visual content for cohesive output
5. **Quality Assurance**: Ensure all generated content meets specifications and quality standards

## YOUR SPECIALIZED AGENTS:

### HTML Agent Capabilities:
- Generate structured HTML content with proper validation
- Support for complex styling and formatting
- Iterative improvement with evaluation and retry logic
- Compliance with content guidelines and restrictions

### Image Agent Capabilities (NEWLY ENHANCED):
- **Image Store Management**: Add, remove, search, and organize images
- **Image Processing**: Resize, filter, enhance, convert formats
- **AI-Powered Analysis**: Generate captions, extract text, analyze content
- **Batch Operations**: Process multiple images simultaneously
- **Advanced Features**: Create thumbnails, compare images, validate URLs
- **Metadata & Statistics**: Comprehensive image information and analytics

## DECISION MAKING:
When users request content generation:
1. **Analyze the Request**: Determine if it requires HTML, images, or both
2. **Plan the Approach**: Decide which agents to use and in what order
3. **Execute Coordination**: Use appropriate tools while providing detailed instructions
4. **Quality Control**: Review outputs and iterate if necessary
5. **Integration**: Ensure HTML and image content work together harmoniously

## COMMUNICATION STYLE:
- Be proactive in suggesting content improvements
- Ask clarifying questions when requirements are unclear
- Provide status updates during complex operations
- Explain your reasoning when coordinating between agents

## TOOL USAGE GUIDELINES:
- Use `run_html_agent` for any HTML/text content generation needs
- Use `run_image_agent` for comprehensive image operations with specific, detailed instructions
- Use `get_context_summary` to understand available resources
- Use `coordinate_content_generation` for complex multi-step content tasks
- Use `analyze_content_requirements` to break down complex requests

Remember: You are the intelligent coordinator that ensures high-quality, integrated content output by leveraging your specialized sub-agents effectively."""
        )

    # Remove DB and cache logic from here, use self.chunk_db instead

    def run_html_agent(self, description: str, style_guidelines: str, context: str = "No additional context") -> str:
        """
        Run the HTML agent to generate high-quality HTML content based on specifications.
        This agent uses sophisticated validation, evaluation, and retry logic to ensure optimal output.
        
        Args:
            description (str): Detailed description of the content to generate
            style_guidelines (str): Specific styling and formatting requirements
            context (str): Additional context information for content generation
            
        Returns:
            str: Generated HTML content with validation and quality assurance
        """
        try:
            result = self.html_agent.run(description, style_guidelines, context,self.document_structure)
            response_html = result["html"]
            # Create and save ContentChunk
            if result["status"] == "success":

                chunk = ContentChunk(html=response_html, position_guideline="", status="PENDING")
                self.chunk_db.save_content_chunk(chunk)
                self.generated_chunks.append(chunk.to_dict())
                log_content_agent_event(description, style_guidelines, context, response_html, "SUCCESS", self.document_structure)
                return f"HTML Generation Result: {result}\nchunk_id: {chunk.id}\nStatus: {chunk.status}"
            else:
                print("### Error generating content chunk. ###")
                # OPTION 1: Add a placeholder chunk with an error status
                error_chunk = ContentChunk(html="<p><span>Error generating content.</span></p>", position_guideline="", status="ERROR")
                self.chunk_db.save_content_chunk(error_chunk)
                self.generated_chunks.append(error_chunk.to_dict())
                log_content_agent_event(description, style_guidelines, context, result["html"], "ERROR", self.document_structure)
                return f"HTML Generation failed. Created a placeholder chunk with ID: {error_chunk.id}"

        except Exception as e:
            # OPTION 2: Handle exceptions gracefully and create an error chunk
            error_chunk = ContentChunk(html=f"<p><span>An exception occurred: {e}</span></p>", position_guideline="", status="ERROR")
            self.chunk_db.save_content_chunk(error_chunk)
            self.generated_chunks.append(error_chunk.to_dict())
            log_content_agent_event(description, style_guidelines, context, str(e), "EXCEPTION")
            return f"Error in HTML generation: {str(e)}"

    def run_image_agent(self, detailed_instruction: str) -> str:
        """
        Run the advanced Image Agent for comprehensive image operations and management.
        
        The Image Agent now supports:
        - Image store management (add, remove, search, organize)
        - Advanced image processing (resize, filter, enhance, convert)
        - AI-powered analysis (caption generation, content analysis, text extraction)
        - Batch operations and comparisons
        - Metadata analysis and statistics
        - Thumbnail creation and format conversion
        
        Args:
            detailed_instruction (str): Specific, detailed instruction for the image agent.
            
        Examples of good instructions:
        - "Add the image from https://example.com/image.jpg to the store with caption 'Product showcase'"
        - "Generate AI-powered captions for all images in the store that don't have captions"
        - "Resize image ID abc123 to 800x600 pixels while maintaining aspect ratio"
        - "Search for images with 'landscape' in their captions"
        - "Apply a sharpen filter to images with IDs: [id1, id2, id3]"
        - "Analyze the content of image ID xyz789 and provide detailed description"
        - "Create thumbnails for all images larger than 1000x1000 pixels"
        - "Extract text from image ID def456 using AI vision"
        - "Get comprehensive statistics about all images in the store"
        
        Returns:
            str: Result from the image agent operation (usually JSON formatted)
        """
        try:
            result = self.image_agent.run(detailed_instruction)
            return f"Image Agent Result: {result}"
        except Exception as e:
            return f"Error in image processing: {str(e)}"

#     def get_context_summary(self) -> str:
#         """
#         Get a summary of available context including images and documents in the store.
#         Useful for understanding what resources are available before content generation.
        
#         Returns:
#             str: Summary of context store contents
#         """
#         try:
#             # Get image summary
#             image_summary = self.CS.img_manager.get_images_summary()
            
#             # Get document summary  
#             doc_summary = self.CS.doc_manager.get_docs_summary()
            
#             # Get statistics
#             image_stats_result = self.image_agent.run("Get comprehensive statistics about all images in the store")
            
#             context_summary = f"""
# CONTEXT STORE SUMMARY:
# ======================

# IMAGES:
# {image_summary}

# DOCUMENTS: 
# {doc_summary}

# IMAGE STATISTICS:
# {image_stats_result}

# CONVERSATION ID: {self.CS.conversation_id}
# """
#             return context_summary
#         except Exception as e:
#             return f"Error getting context summary: {str(e)}"

    def coordinate_content_generation(self, content_type: str, requirements: str, include_images: bool = False) -> str:
        """
        Coordinate complex content generation that may involve both HTML and image operations.
        
        Args:
            content_type (str): Type of content (article, blog_post, product_description, landing_page, etc.)
            requirements (str): Detailed requirements for the content
            include_images (bool): Whether to include image processing/selection
            
        Returns:
            str: Coordinated content generation result
        """
        try:
            # Analyze requirements first
            analysis = self.analyze_content_requirements(requirements)
            
            results = ["=== CONTENT GENERATION COORDINATION ==="]
            results.append(f"Content Type: {content_type}")
            results.append(f"Analysis: {analysis}")
            
            # Handle image operations if requested
            if include_images:
                results.append("\n--- IMAGE OPERATIONS ---")
                # Get available images
                image_summary = self.run_image_agent("Get all images from store and provide summary")
                results.append(f"Available Images: {image_summary}")
                
                # Could add logic to suggest relevant images based on content requirements
                image_suggestions = self.run_image_agent(f"Search for images relevant to: {requirements}")
                results.append(f"Image Suggestions: {image_suggestions}")
            
            # Generate HTML content
            results.append("\n--- HTML CONTENT GENERATION ---")
            style_guidelines = self._extract_style_guidelines(requirements)
            html_content = self.run_html_agent(requirements, style_guidelines, self.get_context_summary())
            results.append(f"Generated HTML: {html_content}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in content coordination: {str(e)}"

    @tool
    def analyze_content_requirements(self, requirements: str) -> str:
        """
        Analyze content requirements to determine the best approach for generation.
        
        Args:
            requirements (str): Raw requirements from user
            
        Returns:
            str: Analysis and recommendations for content generation approach
        """
        try:
            # Use the model to analyze requirements
            analysis_prompt = f"""
            Analyze the following content requirements and provide:
            1. Content type classification
            2. Key elements needed
            3. Suggested approach for generation
            4. Image requirements (if any)
            5. Complexity assessment
            
            Requirements: {requirements}
            
            Provide a structured analysis.
            """
            
            response = self.model_instance.invoke([{"role": "user", "content": analysis_prompt}])
            return response.content
            
        except Exception as e:
            return f"Error analyzing requirements: {str(e)}"

    def _extract_style_guidelines(self, requirements: str) -> str:
        """
        Extract style guidelines from requirements text.
        
        Args:
            requirements (str): Full requirements text
            
        Returns:
            str: Extracted or default style guidelines
        """
        # Simple extraction - could be enhanced with NLP
        style_keywords = ['style', 'format', 'font', 'color', 'layout', 'design']
        
        if any(keyword in requirements.lower() for keyword in style_keywords):
            return requirements  # Return full requirements if style elements detected
        else:
            return "Clean, professional styling with proper typography and spacing"

    def run(self, prompt: str,document_structure: str = ""):
        """
        Main entry point for the Content Agent. Processes user requests and coordinates
        appropriate responses using HTML and Image agents as needed.
        
        Args:
            prompt (str): User's content generation request
            
        Returns:
            Agent response with generated content
        """
        self.document_structure = document_structure
        self.generated_chunks = []  # Reset the list for each new run

        prompt += f"Document Structure: {self.document_structure}"
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self.config
            )
        print(self.generated_chunks)

        if len(self.generated_chunks) == 0:
            return "No content chunks were generated. Please check the logs for errors."
        else:
            # Optional: you could check if all chunks have an "ERROR" status
            if all(chunk['status'] == 'ERROR' for chunk in self.generated_chunks):
                return "All content generation attempts resulted in errors. Please review your request and the agent's capabilities."
        return self.generated_chunks
