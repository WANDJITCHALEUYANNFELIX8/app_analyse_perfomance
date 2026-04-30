from flask import Flask, request, render_template, redirect, url_for, session
import io, base64, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from functools import lru_cache
from models import Student
from controller import process_student

from database import *
from analysis import *

app = Flask(__name__)
app.secret_key = "edustat_secret_2024"
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['SESSION_COOKIE_SECURE'] = True


# ─────────────────────────────
# INIT DB (Render safe)
# ─────────────────────────────
with app.app_context():
    try:
        init_db()
    except Exception as e:
        print("DB non initialisée:", e)
# ─────────────────────────────
# UTIL: image matplotlib -> base64
# ─────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img

@lru_cache(maxsize=10)
def get_all_graphs_cached():
    data = afficher_donnees()

    if len(data) == 0:
        return {}

    data = ajouter_classe(data)
    df_pca, fig0, variance = analyser_pca(data)

    return {
        "hist": fig_b64(graphique_histogramme(data)),
        "classes": fig_b64(graphique_repartition_classes(data)),
        "sexe": fig_b64(graphique_moyenne_sexe(data)),
        "niveau": fig_b64(graphique_moyenne_niveau(data)),
        "corr": fig_b64(graphique_correlations(data)),
        "boxplot": fig_b64(graphique_boxplot(data)),
        "pca": fig_b64(fig0)
    }
    
# ─────────────────────────────
# HOME
# ─────────────────────────────
@app.route('/')
def home():
    nb = count_students()
    stats = {}

    if nb > 0:
        data = ajouter_classe(afficher_donnees())
        stats = {
            "moy_gen": round(data["moyenne"].mean(), 2),
            "classe_top": data["classe"].value_counts().idxmax().title(),
            "nb_excellents": int((data["classe"] == "excellent").sum())
        }

    return render_template("index.html", nb=nb, stats=stats)

# ─────────────────────────────
# FORMULAIRE
# ─────────────────────────────
@app.route('/formulaire')
def formulaire():
    return render_template("formulaire.html")

@app.route('/submit', methods=['POST'])
def submit():
    get_all_graphs_cached.cache_clear()
    form_data = {
        "age": request.form["age"],
        "sexe": request.form["sexe"],
        "etude": request.form["etude"],
        "sommeil": request.form["sommeil"],
        "distraction": request.form["distraction"],
        "env": request.form["env"],
        "assiduite": request.form["assiduite"],
        "ponctualite": request.form["ponctualite"],
        "discipline": request.form["discipline"],
        "tache": request.form["tache"],
        "niveau": request.form["niveau"],
        "moyenne": request.form["moyenne"]
    }

    student = process_student(form_data)
    session["last_student"] = vars(student)

    return redirect(url_for("resultat"))

# ─────────────────────────────
# RESULTAT INDIVIDUEL
# ─────────────────────────────
@app.route('/resultat')
def resultat():

    student_dict = session.get("last_student")
    if not student_dict:
        return redirect(url_for("formulaire"))

    student = Student(**student_dict)

    data = afficher_donnees()

    if "classe" not in data.columns:
        data = ajouter_classe(data)

    classe_finale = "indisponible"
    explication = []

    if len(data) >= 10 and data["classe"].nunique() >= 2:
        try:
            model_clf, scaler, acc = classification_modele(data)

            features = ["etude","sommeil","distraction",
                        "assiduite","ponctualite","discipline","tache"]

            X_new = pd.DataFrame([[float(student_dict[f]) for f in features]], columns=features)

            X_scaled = scaler.transform(X_new)
            classe_ml = model_clf.predict(X_scaled)[0]

            moyenne = float(student.moyenne)

            classe_finale = fusion_classe(moyenne, classe_ml)

            explication = expliquer_classe(moyenne, classe_ml, classe_finale)

        except Exception as e:
            print("Erreur classification :", e)

    moy_gen = round(data["moyenne"].mean(), 2) if len(data) > 0 else "N/A"
    conseils = generer_conseils(student, classe_finale, moy_gen)

    return render_template(
        "individuelle.html",
        student=student,
        classe_predite=classe_finale,
        moyenne_generale=moy_gen,
        conseils=conseils,
        explication=explication
    )

# ─────────────────────────────
# ANALYSE GENERALE
# ─────────────────────────────
@app.route('/generale')
def generale():
    data = afficher_donnees()

    if len(data) == 0:
        return render_template("generale.html", vide=True)

    data = ajouter_classe(data)
    stats = stats_generales(data)
    
    df_pca, fig0, variance = analyser_pca(data)
    
    imgs = get_all_graphs_cached()

    return render_template(
        "generale.html",
        vide=False,
        stats=stats,
        imgs=imgs,
        desc=DESCRIPTIONS
       
    )

# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    
    
