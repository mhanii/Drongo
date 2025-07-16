from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from ContextStore.context_store import ContextStore, ImagePointer
from typing import Annotated, List, Optional, Dict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import sqlite3
import httpx
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import base64
import json
import re
from urllib.parse import urlparse


class State(TypedDict):
    messages: Annotated[List, add_messages]


class ImageAgent:
    def __init__(self, model: ChatGoogleGenerativeAI, checkpoint_path: str) -> None:
        self.connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.model = model
        
        # Define all image processing tools
        self.defined_tools = [
            self.get_images_from_store,
            self.add_image_to_store,
            self.remove_image_from_store,
            self.check_image_validity,
            self.insert_image_to_document,
            self.get_image_details,
            self.search_images_by_caption,
            self.update_image_caption,
            self.generate_image_caption,
            self.resize_image,
            self.apply_image_filter,
            self.enhance_image,
            self.convert_image_format,
            self.get_image_metadata,
            self.validate_image_url_accessibility,
            self.batch_process_images,
            self.compare_images,
            self.extract_text_from_image,
            self.analyze_image_content,
            self.create_image_thumbnail,
            self.get_image_statistics
        ]

        self.agent = create_react_agent(
            model=model,
            tools=self.defined_tools,
            debug=False,
            checkpointer=SqliteSaver(self.connection)
        )
        
        self.config = {"configurable": {"thread_id": "1"}}

    
    def get_images_from_store(self) -> str:
        """
        Retrieves all images currently stored in the image manager.
        Returns a JSON string with a list of dictionaries, where each dictionary represents an image.
        Use this when the user asks to see available images, list all images, or browse the image store.
        
        Returns:
            str: JSON string containing list of image summaries with id, caption, and URL if available.
        """
        try:
            images = self.CS.img_manager.get_all_images()
            result = []
            for img in images:
                img_data = {
                    "image_id": img.get_image_id(),
                    "caption": img.get_caption(),
                    "url": getattr(img, 'url', None)
                }
                result.append(img_data)
            return json.dumps({"status": "success", "images": result, "count": len(result)})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Failed to retrieve images: {str(e)}"})

    
    def add_image_to_store(self, url: str, caption: Optional[str] = None, type: Optional[str] = None) -> str:
        """
        Adds a new image to the image manager from a given URL.
        A caption for the image can optionally be provided.
        Type (format) can also be provided.
        """
        try:
            if not self.CS.img_manager.is_valid_image_url(url):
                return json.dumps({"status": "error", "message": "Invalid image URL format"})
            img_pointer = self.CS.img_manager.add_image_from_url(url, caption or f"Image from {url}", type=type)
            if img_pointer:
                return json.dumps({
                    "status": "success", 
                    "message": "Image added successfully", 
                    "image_id": img_pointer.get_image_id(),
                    "caption": img_pointer.get_caption(),
                    "type": getattr(img_pointer, 'type', None)
                })
            else:
                return json.dumps({"status": "error", "message": "Failed to fetch or store image"})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error adding image: {str(e)}"})

    
    def remove_image_from_store(self, image_id: str) -> str:
        """
        Removes an image from the store by its ID.
        
        Args:
            image_id (str): The ID of the image to remove.
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            success = self.CS.img_manager.remove_image(image_id, delete_from_db=True)
            if success:
                return json.dumps({"status": "success", "message": f"Image {image_id} removed successfully"})
            else:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found or couldn't be removed"})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error removing image: {str(e)}"})

    
    def check_image_validity(self, url: str) -> str:
        """
        Checks if a given URL points to a valid image that can be processed.
        
        Args:
            url (str): The URL of the image to validate.
            
        Returns:
            str: JSON string with validation result.
        """
        try:
            is_valid = self.CS.img_manager.is_valid_image_url(url)
            return json.dumps({"status": "success", "is_valid": is_valid, "url": url})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error validating URL: {str(e)}"})

    
    def insert_image_to_document(self, image_id: str, width: int, height: int, caption: str) -> str:
        """
        Inserts an image into a document with specified dimensions.
        
        Args:
            image_id (str): The ID of the image to insert.
            width (int): The width for the image insertion.
            height (int): The height for the image insertion.
            caption (str): The caption for the image insertion.
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            result = self.CS.img_manager.insert_image_to_document(image_id, width, height, caption)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error inserting image: {str(e)}"})

    
    def get_image_details(self, image_id: str) -> str:
        """
        Retrieves detailed information about a specific image.
        
        Args:
            image_id (str): The ID of the image to get details for.
            
        Returns:
            str: JSON string with detailed image information.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            details = {
                "image_id": img_pointer.get_image_id(),
                "caption": img_pointer.get_caption(),
                "url": getattr(img_pointer, 'url', None)
            }
            
            # Try to get image metadata if available
            try:
                image_data = img_pointer.get_image_data()
                if image_data:
                    image_bytes = base64.b64decode(image_data)
                    img = Image.open(BytesIO(image_bytes))
                    details.update({
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode,
                        "size_bytes": len(image_bytes)
                    })
            except Exception:
                pass
            
            return json.dumps({"status": "success", "details": details})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error getting image details: {str(e)}"})

    
    def search_images_by_caption(self, search_term: str) -> str:
        """
        Searches for images by caption content.
        
        Args:
            search_term (str): The term to search for in image captions.
            
        Returns:
            str: JSON string with matching images.
        """
        try:
            all_images = self.CS.img_manager.get_all_images()
            matching_images = []
            
            for img in all_images:
                caption = img.get_caption().lower()
                if search_term.lower() in caption:
                    matching_images.append({
                        "image_id": img.get_image_id(),
                        "caption": img.get_caption(),
                        "url": getattr(img, 'url', None)
                    })
            
            return json.dumps({
                "status": "success", 
                "search_term": search_term,
                "matches": matching_images, 
                "count": len(matching_images)
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error searching images: {str(e)}"})

    
    def update_image_caption(self, image_id: str, new_caption: str) -> str:
        """
        Updates the caption of an existing image.
        
        Args:
            image_id (str): The ID of the image to update.
            new_caption (str): The new caption for the image.
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            old_caption = img_pointer.get_caption()
            img_pointer.set_caption(new_caption)
            
            return json.dumps({
                "status": "success", 
                "message": f"Caption updated for image {image_id}",
                "old_caption": old_caption,
                "new_caption": new_caption
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error updating caption: {str(e)}"})

    
    def generate_image_caption(self, image_id: str) -> str:
        """
        Generates an AI-powered caption for an image using vision capabilities.
        
        Args:
            image_id (str): The ID of the image to generate a caption for.
            
        Returns:
            str: JSON string with the generated caption.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            # Try to generate caption using the model's vision capabilities
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Create vision prompt
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the content of this image in a detailed, vivid, and specific way. "
                            "List all visible objects, people, actions, settings, and notable details. "
                            "Provide only the caption text without any introductions or explanations."
                        )
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_data,
                        "mime_type": "image/jpeg"
                    }
                ]
            }
            
            response = self.model.invoke([message])
            generated_caption = response.content.strip()
            
            return json.dumps({
                "status": "success",
                "image_id": image_id,
                "generated_caption": generated_caption,
                "original_caption": img_pointer.get_caption()
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error generating caption: {str(e)}"})

    
    def resize_image(self, image_id: str, width: int, height: int, maintain_aspect_ratio: bool = True) -> str:
        """
        Resizes an image to specified dimensions.
        
        Args:
            image_id (str): The ID of the image to resize.
            width (int): Target width in pixels.
            height (int): Target height in pixels.
            maintain_aspect_ratio (bool): Whether to maintain aspect ratio.
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Decode and resize image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            original_size = img.size
            
            if maintain_aspect_ratio:
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
                new_size = img.size
            else:
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                new_size = (width, height)
            
            # Save resized image
            output_buffer = BytesIO()
            img.save(output_buffer, format=img.format or 'JPEG')
            resized_data = base64.b64encode(output_buffer.getvalue()).decode()
            
            # Update image data
            img_pointer.set_image_data(base64.b64decode(resized_data), update_db=True)
            
            return json.dumps({
                "status": "success",
                "message": f"Image {image_id} resized successfully",
                "original_size": original_size,
                "new_size": new_size,
                "aspect_ratio_maintained": maintain_aspect_ratio
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error resizing image: {str(e)}"})

    
    def apply_image_filter(self, image_id: str, filter_type: str) -> str:
        """
        Applies a filter to an image.
        
        Args:
            image_id (str): The ID of the image to filter.
            filter_type (str): Type of filter to apply (blur, sharpen, smooth, contour, edge_enhance).
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Decode image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            
            # Apply filter
            filter_map = {
                "blur": ImageFilter.BLUR,
                "sharpen": ImageFilter.SHARPEN,
                "smooth": ImageFilter.SMOOTH,
                "contour": ImageFilter.CONTOUR,
                "edge_enhance": ImageFilter.EDGE_ENHANCE,
                "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
                "emboss": ImageFilter.EMBOSS,
                "find_edges": ImageFilter.FIND_EDGES
            }
            
            if filter_type not in filter_map:
                available_filters = list(filter_map.keys())
                return json.dumps({
                    "status": "error", 
                    "message": f"Invalid filter type. Available filters: {available_filters}"
                })
            
            filtered_img = img.filter(filter_map[filter_type])
            
            # Save filtered image
            output_buffer = BytesIO()
            filtered_img.save(output_buffer, format=img.format or 'JPEG')
            filtered_data = base64.b64encode(output_buffer.getvalue()).decode()
            
            # Update image data
            img_pointer.set_image_data(base64.b64decode(filtered_data), update_db=True)
            
            return json.dumps({
                "status": "success",
                "message": f"Filter '{filter_type}' applied to image {image_id}",
                "filter_applied": filter_type
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error applying filter: {str(e)}"})

    
    def enhance_image(self, image_id: str, enhancement_type: str, factor: float) -> str:
        """
        Enhances an image by adjusting brightness, contrast, color, or sharpness.
        
        Args:
            image_id (str): The ID of the image to enhance.
            enhancement_type (str): Type of enhancement (brightness, contrast, color, sharpness).
            factor (float): Enhancement factor (1.0 = no change, >1.0 = increase, <1.0 = decrease).
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Decode image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            
            # Apply enhancement
            enhancement_map = {
                "brightness": ImageEnhance.Brightness,
                "contrast": ImageEnhance.Contrast,
                "color": ImageEnhance.Color,
                "sharpness": ImageEnhance.Sharpness
            }
            
            if enhancement_type not in enhancement_map:
                available_enhancements = list(enhancement_map.keys())
                return json.dumps({
                    "status": "error", 
                    "message": f"Invalid enhancement type. Available: {available_enhancements}"
                })
            
            enhancer = enhancement_map[enhancement_type](img)
            enhanced_img = enhancer.enhance(factor)
            
            # Save enhanced image
            output_buffer = BytesIO()
            enhanced_img.save(output_buffer, format=img.format or 'JPEG')
            enhanced_data = base64.b64encode(output_buffer.getvalue()).decode()
            
            # Update image data
            img_pointer.set_image_data(base64.b64decode(enhanced_data), update_db=True)
            
            return json.dumps({
                "status": "success",
                "message": f"Image {image_id} enhanced with {enhancement_type} (factor: {factor})",
                "enhancement_type": enhancement_type,
                "factor": factor
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error enhancing image: {str(e)}"})

    
    def convert_image_format(self, image_id: str, target_format: str) -> str:
        """
        Converts an image to a different format.
        
        Args:
            image_id (str): The ID of the image to convert.
            target_format (str): Target format (JPEG, PNG, GIF, BMP, TIFF).
            
        Returns:
            str: JSON string indicating the status of the operation.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Decode image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            original_format = img.format
            
            # Convert format
            target_format = target_format.upper()
            supported_formats = ["JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"]
            
            if target_format not in supported_formats:
                return json.dumps({
                    "status": "error", 
                    "message": f"Unsupported format. Supported: {supported_formats}"
                })
            
            # Handle RGBA for JPEG
            if target_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = rgb_img
            
            # Save in new format
            output_buffer = BytesIO()
            img.save(output_buffer, format=target_format)
            converted_data = base64.b64encode(output_buffer.getvalue()).decode()
            
            # Update image data
            img_pointer.set_image_data(base64.b64decode(converted_data), update_db=True)
            
            return json.dumps({
                "status": "success",
                "message": f"Image {image_id} converted from {original_format} to {target_format}",
                "original_format": original_format,
                "new_format": target_format
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error converting image format: {str(e)}"})

    
    def get_image_metadata(self, image_id: str) -> str:
        """
        Retrieves comprehensive metadata about an image.
        
        Args:
            image_id (str): The ID of the image to analyze.
            
        Returns:
            str: JSON string with detailed image metadata.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Decode and analyze image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            
            metadata = {
                "image_id": image_id,
                "caption": img_pointer.get_caption(),
                "url": getattr(img_pointer, 'url', None),
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size_bytes": len(image_bytes),
                "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else None,
                "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
            }
            
            # Add EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                metadata["exif_data"] = str(img._getexif())
            
            return json.dumps({"status": "success", "metadata": metadata})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error getting metadata: {str(e)}"})

    
    def validate_image_url_accessibility(self, url: str) -> str:
        """
        Validates if an image URL is accessible and returns detailed information.
        
        Args:
            url (str): The URL to validate.
            
        Returns:
            str: JSON string with validation results.
        """
        try:
            # Check URL format first
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return json.dumps({
                    "status": "error", 
                    "message": "Invalid URL format",
                    "url": url,
                    "accessible": False
                })
            
            # Try to access the URL
            response = httpx.head(url, follow_redirects=True, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            content_length = response.headers.get('content-length')
            
            is_image = any(img_type in content_type for img_type in 
                          ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'])
            
            return json.dumps({
                "status": "success",
                "url": url,
                "accessible": True,
                "is_image": is_image,
                "content_type": content_type,
                "content_length": content_length,
                "status_code": response.status_code
            })
        except httpx.HTTPStatusError as e:
            return json.dumps({
                "status": "error",
                "message": f"HTTP error: {e.response.status_code}",
                "url": url,
                "accessible": False
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error accessing URL: {str(e)}",
                "url": url,
                "accessible": False
            })

    
    def batch_process_images(self, operation: str, image_ids: List[str], **kwargs) -> str:
        """
        Performs batch operations on multiple images.
        
        Args:
            operation (str): Operation to perform (resize, filter, enhance, convert).
            image_ids (List[str]): List of image IDs to process.
            **kwargs: Additional parameters for the operation.
            
        Returns:
            str: JSON string with batch operation results.
        """
        try:
            results = []
            for image_id in image_ids:
                try:
                    if operation == "resize":
                        result = self.resize_image(image_id, kwargs.get('width', 800), 
                                                 kwargs.get('height', 600), 
                                                 kwargs.get('maintain_aspect_ratio', True))
                    elif operation == "filter":
                        result = self.apply_image_filter(image_id, kwargs.get('filter_type', 'blur'))
                    elif operation == "enhance":
                        result = self.enhance_image(image_id, kwargs.get('enhancement_type', 'brightness'), 
                                                  kwargs.get('factor', 1.2))
                    elif operation == "convert":
                        result = self.convert_image_format(image_id, kwargs.get('target_format', 'JPEG'))
                    else:
                        result = json.dumps({"status": "error", "message": f"Unknown operation: {operation}"})
                    
                    results.append({"image_id": image_id, "result": json.loads(result)})
                except Exception as e:
                    results.append({
                        "image_id": image_id, 
                        "result": {"status": "error", "message": str(e)}
                    })
            
            success_count = sum(1 for r in results if r["result"]["status"] == "success")
            
            return json.dumps({
                "status": "success",
                "operation": operation,
                "total_images": len(image_ids),
                "successful": success_count,
                "failed": len(image_ids) - success_count,
                "results": results
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error in batch processing: {str(e)}"})

    
    def compare_images(self, image_id1: str, image_id2: str) -> str:
        """
        Compares two images and provides similarity analysis.
        
        Args:
            image_id1 (str): ID of the first image.
            image_id2 (str): ID of the second image.
            
        Returns:
            str: JSON string with comparison results.
        """
        try:
            img1 = self.CS.img_manager.get_image_by_id(image_id1)
            img2 = self.CS.img_manager.get_image_by_id(image_id2)
            
            if not img1 or not img2:
                return json.dumps({"status": "error", "message": "One or both images not found"})
            
            # Get image data
            data1 = img1.get_image_data()
            data2 = img2.get_image_data()
            
            if not data1 or not data2:
                return json.dumps({"status": "error", "message": "Image data not available"})
            
            # Decode images
            bytes1 = base64.b64decode(data1)
            bytes2 = base64.b64decode(data2)
            pil_img1 = Image.open(BytesIO(bytes1))
            pil_img2 = Image.open(BytesIO(bytes2))
            
            # Basic comparison
            comparison = {
                "image1_id": image_id1,
                "image2_id": image_id2,
                "same_dimensions": pil_img1.size == pil_img2.size,
                "same_format": pil_img1.format == pil_img2.format,
                "same_mode": pil_img1.mode == pil_img2.mode,
                "image1_size": pil_img1.size,
                "image2_size": pil_img2.size,
                "size_difference": abs(len(bytes1) - len(bytes2)),
                "identical_bytes": bytes1 == bytes2
            }
            
            return json.dumps({"status": "success", "comparison": comparison})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error comparing images: {str(e)}"})

    
    def extract_text_from_image(self, image_id: str) -> str:
        """
        Extracts text from an image using AI vision capabilities.
        
        Args:
            image_id (str): The ID of the image to extract text from.
            
        Returns:
            str: JSON string with extracted text.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Create vision prompt for text extraction
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text from this image. "
                            "Provide the text exactly as it appears, maintaining any formatting or structure. "
                            "If no text is visible, respond with 'No text found'."
                        )
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_data,
                        "mime_type": "image/jpeg"
                    }
                ]
            }
            
            response = self.model.invoke([message])
            extracted_text = response.content.strip()
            
            return json.dumps({
                "status": "success",
                "image_id": image_id,
                "extracted_text": extracted_text,
                "has_text": extracted_text.lower() != "no text found"
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error extracting text: {str(e)}"})

    
    def analyze_image_content(self, image_id: str, analysis_type: str = "general") -> str:
        """
        Performs AI-powered analysis of image content.
        
        Args:
            image_id (str): The ID of the image to analyze.
            analysis_type (str): Type of analysis (general, objects, people, scene, colors, emotions).
            
        Returns:
            str: JSON string with analysis results.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Create analysis prompts based on type
            analysis_prompts = {
                "general": "Provide a comprehensive analysis of this image including objects, setting, mood, and notable details.",
                "objects": "Identify and list all objects, items, and things visible in this image.",
                "people": "Analyze any people in this image including their appearance, expressions, poses, and activities.",
                "scene": "Describe the scene, setting, environment, and context of this image.",
                "colors": "Analyze the color palette, dominant colors, and color relationships in this image.",
                "emotions": "Analyze the emotional content, mood, and atmosphere conveyed by this image."
            }
            
            prompt_text = analysis_prompts.get(analysis_type, analysis_prompts["general"])
            
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_data,
                        "mime_type": "image/jpeg"
                    }
                ]
            }
            
            response = self.model.invoke([message])
            analysis_result = response.content.strip()
            
            return json.dumps({
                "status": "success",
                "image_id": image_id,
                "analysis_type": analysis_type,
                "analysis": analysis_result
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error analyzing image: {str(e)}"})

    
    def create_image_thumbnail(self, image_id: str, max_size: int = 150) -> str:
        """
        Creates a thumbnail version of an image.
        
        Args:
            image_id (str): The ID of the image to create a thumbnail for.
            max_size (int): Maximum dimension for the thumbnail.
            
        Returns:
            str: JSON string with thumbnail creation results.
        """
        try:
            img_pointer = self.CS.img_manager.get_image_by_id(image_id)
            if not img_pointer:
                return json.dumps({"status": "error", "message": f"Image {image_id} not found"})
            
            image_data = img_pointer.get_image_data()
            if not image_data:
                return json.dumps({"status": "error", "message": "No image data available"})
            
            # Decode and create thumbnail
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            original_size = img.size
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            thumbnail_size = img.size
            
            # Save thumbnail
            output_buffer = BytesIO()
            img.save(output_buffer, format=img.format or 'JPEG')
            thumbnail_data = base64.b64encode(output_buffer.getvalue()).decode()
            
            # Create new image pointer for thumbnail
            thumbnail_caption = f"Thumbnail of {img_pointer.get_caption()}"
            thumbnail_pointer = self.CS.img_manager.add_image_from_data(
                base64.b64decode(thumbnail_data), 
                thumbnail_caption
            )
            
            return json.dumps({
                "status": "success",
                "original_image_id": image_id,
                "thumbnail_image_id": thumbnail_pointer.get_image_id(),
                "original_size": original_size,
                "thumbnail_size": thumbnail_size,
                "max_dimension": max_size
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error creating thumbnail: {str(e)}"})

    
    def get_image_statistics(self) -> str:
        """
        Provides statistics about all images in the store.
        
        Returns:
            str: JSON string with comprehensive image statistics.
        """
        try:
            all_images = self.CS.img_manager.get_all_images()
            
            if not all_images:
                return json.dumps({
                    "status": "success",
                    "total_images": 0,
                    "message": "No images in store"
                })
            
            # Gather statistics
            total_size = 0
            formats = {}
            dimensions = []
            has_captions = 0
            has_urls = 0
            
            for img in all_images:
                # Count images with captions and URLs
                if img.get_caption():
                    has_captions += 1
                if hasattr(img, 'url') and img.url:
                    has_urls += 1
                
                # Get image data for analysis
                try:
                    image_data = img.get_image_data()
                    if image_data:
                        image_bytes = base64.b64decode(image_data)
                        total_size += len(image_bytes)
                        
                        pil_img = Image.open(BytesIO(image_bytes))
                        
                        # Count formats
                        fmt = pil_img.format or "Unknown"
                        formats[fmt] = formats.get(fmt, 0) + 1
                        
                        # Collect dimensions
                        dimensions.append(pil_img.size)
                except Exception:
                    continue
            
            # Calculate averages
            avg_width = sum(d[0] for d in dimensions) / len(dimensions) if dimensions else 0
            avg_height = sum(d[1] for d in dimensions) / len(dimensions) if dimensions else 0
            
            stats = {
                "total_images": len(all_images),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "images_with_captions": has_captions,
                "images_with_urls": has_urls,
                "formats": formats,
                "average_dimensions": {
                    "width": round(avg_width, 1),
                    "height": round(avg_height, 1)
                } if dimensions else None,
                "analyzed_images": len(dimensions)
            }
            
            return json.dumps({"status": "success", "statistics": stats})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error getting statistics: {str(e)}"})

    def run(self, prompt: str):
        """
        Runs the image agent with the given prompt.
        
        Args:
            prompt (str): The user's prompt/request.
            
        Returns:
            The agent's response after processing the request with available tools.
        """
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=self.config
            )
        return response
