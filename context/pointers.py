from typing import Dict, Optional, Union
import uuid
from Agents.ManagerAgent.DB import sql_client
import base64
from PIL import Image
from io import BytesIO


class ImagePointer:
    def __init__(self,
                filename: str,
                caption: Optional[str] = "", 
                id: Optional[str] = None,
                url: Optional[str] = None, 
                data: Optional[bytes] = None, 
                model: Optional[str] = "models/gemini-1.5-flash-latest",
                type: Optional[str] = None):
        
        self.id = id if id is not None  else str(uuid.uuid4())
        self.caption = caption
        self.filename = filename
        self.type = type
        if url:
            self.url = url

        sql_client.add_image(self.id, self.filename, data, self.caption, self.type)

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "id": self.id,
            "filename" : self.filename,
            "caption": self.caption
        }
        if self.url:
            payload["url"] = self.url
        if self.type:
            payload["type"] = self.type
        return payload
    

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ImagePointer":
        # When loading from DB, we assume image_data is in DB, not in this dict.
        # 'imported=True' signifies it's being loaded, not created fresh.
        return cls(
            id=data["id"],
            filename=data["filename"],
            caption=data["caption"],
            url=data.get("url"),
            type=data.get("type")
        )
    
    def __str__(self):
        url_str = f", url='{self.url}'" if self.url else ""
        return f"Image(id='{self.id}'{url_str}, caption='{self.caption}, filename='{self.filename}')"

    def get_id(self):
        return self.id
    
    def get_data(self):
        # Returns image_data, filename, caption, type
        image = sql_client.get_image(self.id)
        print(image["filename"])
        return image["data"]
    
    def set_data(self, data: bytes, update_db: bool = True):
        """Sets or updates the image data, updating the DB if specified."""
        # self._image_data_cache = image_data # If you implement caching
        if update_db:
            sql_client.add_image(self.id, self.filename, data, self.caption, self.type)

    def delete_from_db(self):
        """Deletes the image data from the database via sql_client."""
        sql_client.delete_image(self.id)


    def get_caption(self) -> str:
        return self.caption
    
    def get_filename(self) -> str:
        return self.filename

    def set_caption(self, caption: str):
        self.caption = caption

    def generate_caption(self):
        image_blob = self.get_data()
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

    def display(self):
        image_blob =  base64.b64decode(self.get_data())
        image = Image.open(BytesIO(image_blob))
        image.show()

class DocumentPointer:
    def __init__(self,
                 data: bytes, # Expect raw bytes
                 filename: str,
                 id: Optional[str] = None,
                 summary : Optional[str] = "",
                 ):
        self.filename = filename
        self.id: str = id if id else str(uuid.uuid4())
        self.summary : str = summary

        sql_client.add_document(self.id, data, self.filename, self.summary)
        print(f"DocumentPointer: New document {self.id} data stored in DB.")


    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "filename": self.filename,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DocumentPointer":
        # Assumes doc_data is in DB, not in this dict.
        return cls(
            id=data["id"],
            filename=data["filename"],
            caption=data["summary"],
        )

    def __str__(self):
        return f"Document(id='{self.id}', summary='{self.caption}')"

    def get_id(self) -> str:
        return self.id

    def get_filename(self) -> str:
        return self.filename
    
    def get_data(self) -> Optional[bytes]:
        # Returns doc_data, filename, summary
        return sql_client.get_document(self.id)["data"]

    def set_data(self, data: bytes, update_db: bool = True):
        """Sets or updates the document data, updating the DB if specified."""
        if update_db:
            sql_client.add_document(self.id, data, self.filename, self.summary)

    def delete_from_db(self):
        """Deletes the document data from the database via sql_client."""
        sql_client.delete_document(self.id)

    def get_summary(self) -> str:
        return self.summary

    def set_summary(self, summary: str):
        self.summary = summary
