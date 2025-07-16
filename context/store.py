from collections import deque
from typing import TypeVar, Generic, Callable, Dict, Optional, List, Union, Protocol

from database import sql_client
from langgraph.graph.message import add_messages  # if still needed
from langchain_google_genai import ChatGoogleGenerativeAI  # if still needed

from .pointers import ImagePointer, DocumentPointer  # adjust import paths
from .managers import PointerManager
from collections import deque



class ContextStore:
    """
    ContextStore wires together two PointerManagers:
      - image_mgr for ImagePointer
      - doc_mgr   for DocumentPointer
    All LRU evictions and DB writes happen in the managers.
    """
    def __init__(self, max_window: int = 10):
        # ImagePointer manager
        self.image_mgr = PointerManager[ImagePointer](
            maxlen=max_window,
            add_to_db=lambda img: sql_client.add_image(
                img.get_id(),
                img.get_filename(),
                img.get_data()[0] if isinstance(img.get_data(), tuple) else img.get_data(),
                img.get_caption(),
                getattr(img, 'type', None)
            ),
            delete_from_db=lambda img: sql_client.delete_image(img.get_id())
        )

        # DocumentPointer manager
        self.doc_mgr = PointerManager[DocumentPointer](
            maxlen=max_window,
            add_to_db=lambda doc: sql_client.add_document(
                doc.get_id(),
                doc.get_data()[0] if isinstance(doc.get_data(), tuple) else doc.get_data(),
                doc.get_filename(),
                doc.get_summary()
            ),
            delete_from_db=lambda doc: sql_client.delete_document(doc.get_id())
        )

    # ---- Image APIs ----

    def add_image(self, img: ImagePointer, to_front: bool = False) -> None:
        """Add or reinsert image into context (MRU behavior)."""
        self.image_mgr.add(img, to_front=to_front)

    def get_image(self, image_id: str) -> Optional[ImagePointer]:
        return self.image_mgr.get(image_id)

    def recent_images(self) -> List[ImagePointer]:
        return self.image_mgr.recent()

    def remove_image(self, image_id: str, delete_from_db: bool = True) -> bool:
        return self.image_mgr.remove(image_id, delete_from_db=delete_from_db)

    # ---- Document APIs ----

    def add_document(self, doc: DocumentPointer, to_front: bool = False) -> None:
        """Add or reinsert document into context (MRU behavior)."""
        self.doc_mgr.add(doc, to_front=to_front)

    def get_document(self, doc_id: str) -> Optional[DocumentPointer]:
        return self.doc_mgr.get(doc_id)

    def recent_documents(self) -> List[DocumentPointer]:
        return self.doc_mgr.recent()

    def remove_document(self, doc_id: str, delete_from_db: bool = True) -> bool:
        return self.doc_mgr.remove(doc_id, delete_from_db=delete_from_db)
