import psycopg2
import os

DATABASE_URL = os.environ.get("DATABASE_URL")

def connect():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS student (
        id SERIAL PRIMARY KEY,
        age INT,
        sexe TEXT,
        etude FLOAT,
        sommeil FLOAT,
        distraction FLOAT,
        env FLOAT,
        assiduite FLOAT,
        ponctualite FLOAT,
        discipline FLOAT,
        tache FLOAT,
        niveau TEXT,
        moyenne FLOAT
    )
    """)

    conn.commit()
    conn.close()

def insert_student(data_tuple):
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO student (
            age, sexe, etude, sommeil, distraction,
            env, assiduite, ponctualite, discipline,
            tache, niveau, moyenne
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, data_tuple)

    conn.commit()
    conn.close()

def count_students():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM student")
    nb = cur.fetchone()[0]
    conn.close()
    return nb
