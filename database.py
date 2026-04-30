import os
import psycopg2
from psycopg2 import pool

DATABASE_URL = os.environ.get("DATABASE_URL")

# ── pool de connexions (3 min, 10 max) ───────────────────────────
# évite d'ouvrir une nouvelle connexion à chaque requête
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise Exception("DATABASE_URL non définie dans les variables d'environnement.")
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
            sslmode="require"
        )
    return _pool

def connect():
    """Retourne une connexion depuis le pool."""
    return get_pool().getconn()

def release(conn):
    """Remet la connexion dans le pool."""
    get_pool().putconn(conn)

def init_db():
    """Crée la table si elle n'existe pas."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS student (
                id          SERIAL PRIMARY KEY,
                age         INT,
                sexe        TEXT,
                etude       FLOAT,
                sommeil     FLOAT,
                distraction FLOAT,
                env         FLOAT,
                assiduite   FLOAT,
                ponctualite FLOAT,
                discipline  FLOAT,
                tache       FLOAT,
                niveau      TEXT,
                moyenne     FLOAT
            )
        """)
        conn.commit()
    finally:
        release(conn)

def insert_student(data_tuple):
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO student (
                age, sexe, etude, sommeil, distraction,
                env, assiduite, ponctualite, discipline,
                tache, niveau, moyenne
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, data_tuple)
        conn.commit()
    finally:
        release(conn)

def count_students():
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM student")
        return cur.fetchone()[0]
    finally:
        release(conn)

def get_all_students():
    conn = connect()
    try:
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM student", conn)
        return df
    finally:
        release(conn)
