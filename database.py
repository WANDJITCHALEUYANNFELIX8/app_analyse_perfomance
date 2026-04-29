import psycopg2
import os
import sqlite3

DATABASE_URL = os.environ.get("DATABASE_URL")


def connect():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect("student.db")


def create_table():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student (
            id SERIAL PRIMARY KEY,
            age INT,
            sexe VARCHAR(2),
            etude FLOAT,
            sommeil FLOAT,
            distraction FLOAT,
            env FLOAT,
            assiduite FLOAT,
            ponctualite FLOAT,
            discipline FLOAT,
            tache FLOAT,
            niveau VARCHAR(10),
            moyenne FLOAT
        )
    """)

    conn.commit()
    conn.close()

def insert_student(data_tuple):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO student (
            age, sexe, etude, sommeil, distraction,
            env, assiduite, ponctualite, discipline,
            tache, niveau, moyenne
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data_tuple)

    conn.commit()
    conn.close()



def count_students():
 
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM student")
    nb = cursor.fetchone()[0]
    conn.close()
    return nb
 
def get_all_students():
    
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM student")
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]
    
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
    
    
    
    
    
    
    
    
    
    
