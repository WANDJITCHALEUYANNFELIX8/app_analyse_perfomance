import random
import psycopg2
import os

DATABASE_URL = os.environ.get("DATABASE_URL")

def connect():
    return psycopg2.connect(DATABASE_URL)

def clear_table():
    conn = connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM student")
    conn.commit()
    conn.close()
    print("🧹 Table vidée")

def generate_profile(group):
    """Crée un profil cohérent selon le niveau"""

    if group == "faible":
        etude = random.uniform(0, 4)
        assiduite = random.uniform(0, 5)
        discipline = random.uniform(0, 5)
        tache = random.uniform(0, 5)
        distraction = random.uniform(6, 10)
        sommeil = random.uniform(3, 8)

    elif group == "moyen":
        etude = random.uniform(3, 7)
        assiduite = random.uniform(4, 7)
        discipline = random.uniform(4, 7)
        tache = random.uniform(4, 7)
        distraction = random.uniform(3, 7)
        sommeil = random.uniform(5, 9)

    else:  # excellent
        etude = random.uniform(6, 10)
        assiduite = random.uniform(7, 10)
        discipline = random.uniform(7, 10)
        tache = random.uniform(7, 10)
        distraction = random.uniform(0, 3)
        sommeil = random.uniform(6, 10)

    # 🎯 moyenne réaliste
    moyenne = (
        etude * 1.5 +
        assiduite * 1.2 +
        discipline * 1.2 +
        tache * 1.0 +
        sommeil * 0.5 -
        distraction * 1.3
    )

    moyenne = max(0, min(20, moyenne + random.uniform(-1.5, 1.5)))

    return etude, sommeil, distraction, assiduite, discipline, tache, moyenne


def seed_students(n=100):
    conn = connect()
    cur = conn.cursor()

    for _ in range(n):

        r = random.random()

        # 📊 distribution réaliste
        if r < 0.3:
            group = "faible"
        elif r < 0.8:
            group = "moyen"
        else:
            group = "excellent"

        etude, sommeil, distraction, assiduite, discipline, tache, moyenne = generate_profile(group)

        sexe = random.choice(["M", "F"])
        niveau = random.choice(["L1", "L2", "L3", "M1", "M2"])

        cur.execute("""
            INSERT INTO student (
                age, sexe, etude, sommeil, distraction,
                env, assiduite, ponctualite, discipline,
                tache, niveau, moyenne
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            random.randint(18, 30),
            sexe,
            etude,
            sommeil,
            distraction,
            random.uniform(3, 10),
            assiduite,
            random.uniform(3, 10),
            discipline,
            tache,
            niveau,
            moyenne
        ))

    conn.commit()
    conn.close()
    print(f"✅ {n} étudiants cohérents insérés")


if __name__ == "__main__":
    clear_table()
    seed_students(100)
