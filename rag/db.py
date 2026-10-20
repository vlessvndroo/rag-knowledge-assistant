import psycopg
from pgvector.psycopg import register_vector

from rag.config import DATABASE_URL


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn
