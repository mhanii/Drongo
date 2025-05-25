from typing import TypedDict, Annotated, Dict, Optional , List
from langgraph.graph.message import add_messages # Ensure this is correctly used or messages are plain lists
import uuid
from Agents.ManagerAgent.DB import sql_client
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
import httpx
from PIL import Image
from io import BytesIO

class ImagePointer:
    def __init__(self, caption: str, img_url: Optional[str] = None, image_data: Optional[bytes] = None ,model: str = "models/gemini-1.5-flash-latest", imported: bool = False):
        self.image_id = str(uuid.uuid4())
        self.caption = caption
        if img_url:
            self.url = img_url
        if not imported:
            sql_client.add_image(self.image_id, image_data)

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "image_id": self.image_id,
            "caption": self.caption
        }
        if self.url:
            payload["url"] = self.url
        return payload
    

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ImagePointer":
        # When loading from DB, we assume image_data is in DB, not in this dict.
        # 'imported=True' signifies it's being loaded, not created fresh.
        return cls(
            image_id=data["image_id"],
            caption=data["caption"],
            img_url=data.get("url"),
            imported=True # Key: doesn't re-add to DB on instantiation
        )
    
    def __str__(self):
        url_str = f", url='{self.url}'" if self.url else ""
        return f"Image(image_id='{self.image_id}'{url_str}, caption='{self.caption}')"

    def get_image_id(self):
        return self.image_id
    
    def get_image_data(self) -> bytes | None:
        return sql_client.get_image(self.image_id)
    
    def set_image_data(self, image_data: bytes, update_db: bool = True):
        """Sets or updates the image data, updating the DB if specified."""
        # self._image_data_cache = image_data # If you implement caching
        if update_db:
            sql_client.add_image(self.image_id, image_data) # add_image also updates

    def delete_from_db(self):
        """Deletes the image data from the database via sql_client."""
        sql_client.delete_image(self.image_id)


    def get_caption(self) -> str:
        return self.caption

    def set_caption(self, caption: str):
        self.caption = caption

    def generate_caption(self):
        image_blob = self.get_image()
        message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe the content of this image in a very detailed, vivid, and specific way. "
                                "List all visible objects, people, actions, settings, and notable details. "
                                "Do not include any introductions or explanations-respond only with the caption text."
                            ),
                        },
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image_blob,
                            "mime_type": "image/jpeg",
                        },
                    ],
                }
        response = self.model.invoke([message])
        print(response.text())

        return response.text()
    def delete_caption(self):
        self.caption = ""

    def display_image(self):
        image_blob =  base64.b64decode(self.get_image())
        image = Image.open(BytesIO(image_blob))
        image.show()

class DocumentPointer:
    def __init__(self,
                 caption: str,
                 doc_id: Optional[str] = None,
                 doc_data: Optional[bytes] = None, # Expect raw bytes
                 imported: bool = False):

        self.doc_id: str = doc_id if doc_id else str(uuid.uuid4())
        self.caption: str = caption

        if not imported and doc_data: # Only add to DB if data is provided and it's not an import
            sql_client.add_document(self.doc_id, doc_data)
            print(f"DocumentPointer: New document {self.doc_id} data stored in DB.")
        elif not imported:
            # If no data provided for a new doc, we could still register it with NULL data
            # sql_client.add_document(self.doc_id, None) # If your DB schema/logic supports it
            print(f"DocumentPointer: New document {self.doc_id} created without initial data in DB.")


    def to_dict(self) -> Dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "caption": self.caption
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DocumentPointer":
        # Assumes doc_data is in DB, not in this dict.
        return cls(
            doc_id=data["doc_id"],
            caption=data["caption"],
            imported=True # Doesn't re-add to DB
        )

    def __str__(self):
        return f"Document(doc_id='{self.doc_id}', caption='{self.caption}')"

    def get_doc_id(self) -> str:
        return self.doc_id

    def get_doc_data(self) -> Optional[bytes]:
        """Retrieves document data from the database via sql_client."""
        return sql_client.get_document(self.doc_id)

    def set_doc_data(self, doc_data: bytes, update_db: bool = True):
        """Sets or updates the document data, updating the DB if specified."""
        if update_db:
            sql_client.add_document(self.doc_id, doc_data)

    def delete_from_db(self):
        """Deletes the document data from the database via sql_client."""
        sql_client.delete_document(self.doc_id)

    def get_caption(self) -> str:
        return self.caption

    def set_caption(self, caption: str):
        self.caption = caption

    

class ImageManager:
    def __init__(self, images_dict: Optional[Dict[str, ImagePointer]] = None) -> None:
        self.images: Dict[str, ImagePointer] = images_dict if images_dict is not None else {}

    def add_pointer(self, img_pointer: ImagePointer) -> None:
        if img_pointer.get_image_id() in self.images:
            print(f"ImageManager: Warning - Image with ID {img_pointer.get_image_id()} already managed. Overwriting pointer in manager.")
        self.images[img_pointer.get_image_id()] = img_pointer
        print(f"ImageManager: Added/Updated image pointer for ID {img_pointer.get_image_id()}")

    def add_image_from_url(self, url: str, caption: Optional[str] = "") -> Optional[ImagePointer]:
        if not self.is_valid_image_url(url): # Check first if it even looks like an image URL
            print(f"ImageManager: Invalid or non-image URL: {url}")
            # Optionally try to download anyway if strict check is too restrictive
            # For now, let's be strict.
            # return None 
        
        try:
            print(f"ImageManager: Attempting to fetch image from URL: {url}")
            response = httpx.get(url, follow_redirects=True, timeout=10)
            response.raise_for_status()
            image_data: bytes = response.content

            # Create ImagePointer. Its __init__ will handle DB storage (imported=False).
            img_pointer = ImagePointer(caption=caption or f"Image from {url}", img_url=url, image_data=image_data, imported=False)
            self.add_pointer(img_pointer)
            print(f"ImageManager: Image from {url} successfully fetched, stored (ID: {img_pointer.get_image_id()}), and added to manager.")
            return img_pointer
        except httpx.HTTPStatusError as e:
            print(f"ImageManager: HTTP error fetching image from URL {url}: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"ImageManager: Network error fetching image from URL {url}: {e}")
        except Exception as e:
            print(f"ImageManager: An unexpected error occurred while adding image from {url}: {e}")
        return None

    def add_image_from_data(self, image_data: bytes, caption: str, url: Optional[str] = None) -> ImagePointer:
        # Create ImagePointer. Its __init__ will handle DB storage (imported=False).
        img_pointer = ImagePointer(caption=caption, image_data=image_data, img_url=url, imported=False)
        self.add_pointer(img_pointer)
        print(f"ImageManager: Image from data successfully stored (ID: {img_pointer.get_image_id()}) and added to manager.")
        return img_pointer

    def get_image_by_id(self, image_id: str) -> Optional[ImagePointer]:
        return self.images.get(image_id)

    def remove_image(self, image_id: str, delete_from_db: bool = True) -> bool:
        img_pointer = self.images.pop(image_id, None)
        if img_pointer:
            if delete_from_db:
                img_pointer.delete_from_db()
                print(f"ImageManager: Image data for {image_id} deleted from DB.")
            print(f"ImageManager: Removed image pointer {image_id} from manager.")
            return True
        print(f"ImageManager: Image {image_id} not found in manager for removal.")
        return False

    def get_all_images(self) -> List[ImagePointer]:
        return list(self.images.values())

    def get_all_image_ids(self) -> List[str]:
        return list(self.images.keys())
    
    def get_images_summary(self) -> str:
        if not self.images:
            return "No images in manager."
        return "\n".join([str(img) for img in self.images.values()])
    
    @staticmethod
    def is_valid_image_url(url):
        try:
            response = httpx.head(url, follow_redirects=True, timeout=5)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                return content_type.startswith('image')
            return False
        except httpx.RequestError:
            return False

    def insert_image_to_document(self, image_id: str, width: int, height: int, caption: str) -> dict:
        """
        Inserts an image into a document.
        Args:
            image_id (str): The ID of the image to insert.
            width (int): The width of the image.
            height (int): The height of the image.
            caption (str): The caption of the image.
        Returns:
            A dictionary indicating the status of the operation (e.g., {"status": "success", "message": "Image inserted"}, 
            {"status": "error", "message": "Image not found"}, {"status": "warning", "message": "Document not found"}).
        Use this tool when the user explicitly asks to insert an image into a document. You must provide the image ID, width, height, and caption.
        """
        image_pointer = self.get_image_by_id(image_id)
        image_pointer.display_image()
        
        return {"status": "success", "message": "Image inserted"}
            
    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {"images": [img.to_dict() for img in self.images.values()]}

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, str]]]) -> "ImageManager":
        manager = cls()
        for img_data_dict in data.get("images", []):
            # ImagePointer.from_dict rehydrates pointer without accessing DB for data
            img_pointer = ImagePointer.from_dict(img_data_dict)
            manager.add_pointer(img_pointer) # Adds to manager's dict
        return manager

    
        
class DocManager:
    def __init__(self, docs_dict: Optional[Dict[str, DocumentPointer]] = None) -> None:
        self.docs: Dict[str, DocumentPointer] = docs_dict if docs_dict is not None else {}

    def add_pointer(self, doc_pointer: DocumentPointer):
        if doc_pointer.get_doc_id() in self.docs:
            print(f"DocManager: Warning - Document with ID {doc_pointer.get_doc_id()} already managed. Overwriting pointer in manager.")
        self.docs[doc_pointer.get_doc_id()] = doc_pointer
        print(f"DocManager: Added/Updated document pointer for ID {doc_pointer.get_doc_id()}")

    def add_document_from_data(self, doc_data: bytes, caption: str) -> DocumentPointer:
        # DocumentPointer.__init__ handles DB storage (imported=False)
        doc_pointer = DocumentPointer(caption=caption, doc_data=doc_data, imported=False)
        self.add_pointer(doc_pointer)
        print(f"DocManager: Document from data successfully stored (ID: {doc_pointer.get_doc_id()}) and added to manager.")
        return doc_pointer

    def get_doc_by_id(self, doc_id: str) -> Optional[DocumentPointer]:
        return self.docs.get(doc_id)

    def remove_document(self, doc_id: str, delete_from_db: bool = True) -> bool:
        doc_pointer = self.docs.pop(doc_id, None)
        if doc_pointer:
            if delete_from_db:
                doc_pointer.delete_from_db()
                print(f"DocManager: Document data for {doc_id} deleted from DB.")
            print(f"DocManager: Removed document pointer {doc_id} from manager.")
            return True
        print(f"DocManager: Document {doc_id} not found in manager for removal.")
        return False

    def get_all_docs(self) -> List[DocumentPointer]:
        return list(self.docs.values())
    
    def get_all_doc_ids(self) -> List[str]:
        return list(self.docs.keys())

    def get_docs_summary(self) -> str:
        if not self.docs:
            return "No documents in manager."
        return "\n".join([str(doc) for doc in self.docs.values()])

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {"docs": [doc.to_dict() for doc in self.docs.values()]}

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, str]]]) -> "DocManager":
        manager = cls()
        for doc_data_dict in data.get("docs", []):
            doc_pointer = DocumentPointer.from_dict(doc_data_dict)
            manager.add_pointer(doc_pointer)
        return manager

class ContextStore:
    def __init__(self,
                 conversation_id: str,
                 img_manager: Optional[ImageManager] = None,
                 doc_manager: Optional[DocManager] = None) -> None:
        self.conversation_id = conversation_id
        self.img_manager: ImageManager = img_manager if img_manager is not None else ImageManager()
        self.doc_manager: DocManager = doc_manager if doc_manager is not None else DocManager()

    def to_dict(self) -> Dict[str, any]:
        return {
            "conversation_id": self.conversation_id,
            "image_manager": self.img_manager.to_dict(),
            "doc_manager": self.doc_manager.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "ContextStore":
        # Initialize db first if not already done (e.g. if running this as standalone script)
        # sql_client.init_image_db() # Already called at module import of sql_client
        # sql_client.init_doc_db()   # Already called at module import of sql_client
        return cls(
            conversation_id=data["conversation_id"],
            img_manager=ImageManager.from_dict(data.get("image_manager", {"images": []})),
            doc_manager=DocManager.from_dict(data.get("doc_manager", {"docs": []}))
        )

    def get_image(self, image_id: str) -> Optional[ImagePointer]:
        return self.img_manager.get_image_by_id(image_id)

    def get_document(self, doc_id: str) -> Optional[DocumentPointer]:
        return self.doc_manager.get_doc_by_id(doc_id)

    def add_image_url_to_context(self, url: str, caption: Optional[str] = "") -> Optional[ImagePointer]:
        """Downloads image from URL, stores it, and adds to context's image manager."""
        return self.img_manager.add_image_from_url(url, caption)
    
    def add_image_data_to_context(self, image_data: bytes, caption: str, url: Optional[str] = None) -> ImagePointer:
        """Adds image from byte data, stores it, and adds to context's image manager."""
        return self.img_manager.add_image_from_data(image_data, caption, url)

    def add_document_data_to_context(self, doc_data: bytes, caption: str) -> DocumentPointer:
        """Adds document from byte data, stores it, and adds to context's document manager."""
        return self.doc_manager.add_document_from_data(doc_data, caption)

    def __str__(self):
        img_summary = self.img_manager.get_images_summary()
        doc_summary = self.doc_manager.get_docs_summary()
        return (f"ContextStore(conversation_id='{self.conversation_id}')\n"
                f"--- Images ---\n{img_summary}\n"
                f"--- Documents ---\n{doc_summary}")
