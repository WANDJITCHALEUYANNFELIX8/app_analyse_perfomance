import random
from database import connect, init_db

# ─────────────────────────────────────────────
# 🔥 RESET TABLE (VIDER LA BASE)
# ─────────────────────────────────────────────
def clear_table():
    conn = connect()
    cur = conn.cursor()

    cur.execute("DELETE FROM student")  # vide la table
    conn.commit()
    conn.close()

    print("🧹 Table student vidée")


# ─────────────────────────────────────────────
# 🎯 GÉNÉRATION COHÉRENTE
# ─────────────────────────────────────────────
def generate_student():
    etude = random.uniform(0, 10)
    sommeil = random.uniform(3, 10)
    distraction = random.uniform(0, 10)
    assiduite = random.uniform(0, 10)
    ponctualite = random.uniform(0, 10)
    discipline = random.uniform(0, 10)
    tache = random.uniform(0, 10)

    # 🔥 logique métier (IMPORTANT)
    moyenne = (
        etude * 1.4 +
        assiduite * 1.0 +
        discipline * 0.8 +
        ponctualite * 0.6 +
        sommeil * 0.3 -
        distraction * 1.2 +
        random.uniform(-1, 1)
    )

    moyenne = max(0, min(20, moyenne))

    return (
        random.randint(15, 40),
        random.choice(["M", "F"]),
        round(etude, 2),
        round(sommeil, 2),
        round(distraction, 2),
        round(random.uniform(0, 10), 2),
        round(assiduite, 2),
        round(ponctualite, 2),
        round(discipline, 2),
        round(tache, 2),
        random.choice(["L1", "L2", "L3", "M1", "M2"]),
        round(moyenne, 2)
    )


# ─────────────────────────────────────────────
# 📥 INSERTION
# ─────────────────────────────────────────────
def seed_students(n=100):
    conn = connect()
    cur = conn.cursor()

    for _ in range(n):
        student = generate_student()

        cur.execute("""
            INSERT INTO student (
                age, sexe, etude, sommeil, distraction,
                env, assiduite, ponctualite, discipline,
                tache, niveau, moyenne
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, student)

    conn.commit()
    conn.close()

    print(f"✅ {n} étudiants insérés")


# ─────────────────────────────────────────────
# 🚀 EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    clear_table()        # 🔥 vide la base avant
    seed_students(100)   # 🔥 remplit proprement
