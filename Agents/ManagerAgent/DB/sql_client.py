import sqlite3
import os

IMAGE_DB_PATH = "data/image.sqlite"
DOC_DB_PATH = "data/doc.sqlite"

def get_db_connection(db_path):
    return sqlite3.connect(db_path, check_same_thread=False)

def init_image_db():
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            data BLOB,
            filename TEXT,
            caption TEXT,
            type TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_doc_db():
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            data BLOB,
            filename TEXT,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize databases when module is loaded
init_image_db()
init_doc_db()

def add_image(image_id: str, filename: str, image_data: bytes = None, caption: str = None, type: str = None):
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO images (id, data, filename, caption, type) VALUES (?, ?, ?, ?, ?)", (image_id, image_data, filename, caption, type))
    conn.commit()
    conn.close()

def get_image(image_id: str):
    conn = get_db_connection(IMAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_image(image_id: str):
    conn = get_db_connection(IMAGE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()

def add_document(doc_id: str, doc_data: bytes, filename: str = None, summary: str = None):
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO documents (id, data, filename, summary) VALUES (?, ?, ?, ?)", (doc_id, doc_data, filename, summary))
    conn.commit()
    conn.close()

def get_document(doc_id: str):
    conn = get_db_connection(DOC_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_document(doc_id: str):
    conn = get_db_connection(DOC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

