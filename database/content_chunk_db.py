import sqlite3
import uuid
from typing import List, Optional

class ContentChunk:
    """
    Represents a generated content chunk with an id, html, position guideline, and status.
    Images attribute is reserved for future use.
    """
    def __init__(self, html: str, position_guideline: str = "", id: str = None, status: str = "PENDING"):
        self.id = id or str(uuid.uuid4())
        self.html = html
        self.position_guideline = position_guideline or ""
        self.status = status  # 'APPLIED', 'PENDING', 'CANCELED'
        self.images = []  # Reserved for future use

    def to_dict(self):
        return {
            "status": self.status,
            "chunk_id": self.id,
            "html": self.html,

        }

    @classmethod
    def from_dict(cls, data: dict):
        chunk = cls(
            id=data.get("id"),
            html=data.get("html", ""),
            position_guideline=data.get("position_guideline", ""),
            status=data.get("status", "PENDING")
        )
        chunk.images = data.get("images", [])
        return chunk

    def __str__(self):
        return f"ContentChunk(id = {self.id}, html = {self.html}, images={self.images}, status: {self.status})"

class ContentChunkDB:
    def __init__(self, connection: sqlite3.Connection, max_cache_size: int = 10):
        self.connection = connection
        self.max_cache_size = max_cache_size
        self.recent_chunks: List[ContentChunk] = []
        self._create_content_chunk_table()

    def _create_content_chunk_table(self):
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_chunks (
                id TEXT PRIMARY KEY,
                html TEXT NOT NULL,
                position_guideline TEXT,
                status TEXT DEFAULT 'PENDING'
                -- images TEXT  -- Reserved for future use
            )
        ''')
        self.connection.commit()

    def save_content_chunk(self, chunk: ContentChunk):
        cursor = self.connection.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO content_chunks (id, html, position_guideline, status) VALUES (?, ?, ?, ?)',
            (chunk.id, chunk.html, chunk.position_guideline, chunk.status)
        )
        self.connection.commit()
        self._add_to_cache(chunk)

    def load_content_chunk(self, id: str) -> Optional[ContentChunk]:
        cursor = self.connection.cursor()
        cursor.execute('SELECT id, html, position_guideline, status FROM content_chunks WHERE id = ?', (id,))
        row = cursor.fetchone()
        if row:
            return ContentChunk(id=row[0], html=row[1], position_guideline=row[2], status=row[3])
        return None

    def _add_to_cache(self, chunk: ContentChunk):
        self.recent_chunks.insert(0, chunk)
        if len(self.recent_chunks) > self.max_cache_size:
            self.recent_chunks = self.recent_chunks[:self.max_cache_size]

    def get_recent_chunks(self) -> List[ContentChunk]:
        return self.recent_chunks 
    
    