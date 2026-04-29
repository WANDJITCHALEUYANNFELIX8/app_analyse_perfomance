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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sexe TEXT,
            etude REAL,
            sommeil REAL,
            distraction REAL,
            env INTEGER,
            assiduite INTEGER,
            ponctualite INTEGER,
            discipline INTEGER,
            tache INTEGER,
            niveau TEXT,
            moyenne REAL
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
    
