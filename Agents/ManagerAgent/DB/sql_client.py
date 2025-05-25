import sqlite3
import os

IMAGE_DB_PATH = "image.sqlite"
DOC_DB_PATH = "doc.sqlite"

def get_db_connection(db_path):
    return sqlite3.connect(db_path, check_same_thread=False)

def init_image_db():
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            image_data BLOB
        )
    """)
    conn.commit()
    conn.close()

def init_doc_db():
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doc_data BLOB
        )
    """)
    conn.commit()
    conn.close()

# Initialize databases when module is loaded
init_image_db()
init_doc_db()

def add_image(image_id: str, image_data: bytes):
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO images (image_id, image_data) VALUES (?, ?)", (image_id, image_data))
    conn.commit()
    conn.close()

def get_image(image_id: str) -> bytes | None:
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_data FROM images WHERE image_id = ?", (image_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def delete_image(image_id: str):
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
    conn.commit()
    conn.close()

def add_document(doc_id: str, doc_data: bytes):
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO documents (doc_id, doc_data) VALUES (?, ?)", (doc_id, doc_data))
    conn.commit()
    conn.close()

def get_document(doc_id: str) -> bytes | None:
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_data FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def delete_document(doc_id: str):
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()

