import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from flask import Flask, request, render_template, redirect, url_for, session
from models import Student
from controller import process_student
from database import init_db, count_students
from analysis import (
    afficher_donnees, ajouter_classe, stats_generales,
    analyser_pca, classification_modele, fusion_classe,
    expliquer_classe, generer_conseils, DESCRIPTIONS,
    graphique_histogramme, graphique_repartition_classes,
    graphique_moyenne_sexe, graphique_moyenne_niveau,
    graphique_correlations, graphique_boxplot
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "edustat_secret_2024")

# ── cookies de session compatibles tous navigateurs/téléphones ────
# SESSION_COOKIE_SECURE doit être False en HTTP (Render HTTP) ou
# activé uniquement si HTTPS est garanti
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['SESSION_COOKIE_SECURE']   = os.environ.get("HTTPS", "false") == "true"
app.config['SESSION_COOKIE_HTTPONLY'] = True

# ── init base de données ──────────────────────────────────────────
with app.app_context():
    try:
        init_db()
        print("✅ Base de données initialisée.")
    except Exception as e:
        print(f"⚠️ Erreur init DB : {e}")

# ── utilitaire figure → base64 ─────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img

# ─────────────────────────────────────────────────────────────────
# ACCUEIL
# ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    nb    = count_students()
    stats = {}
    if nb > 0:
        data = ajouter_classe(afficher_donnees())
        stats = {
            "moy_gen":       round(data["moyenne"].mean(), 2),
            "classe_top":    data["classe"].value_counts().idxmax().title(),
            "nb_excellents": int((data["classe"] == "excellent").sum())
        }
    return render_template("index.html", nb=nb, stats=stats)

# ─────────────────────────────────────────────────────────────────
# FORMULAIRE
# ─────────────────────────────────────────────────────────────────
@app.route('/formulaire')
def formulaire():
    return render_template("formulaire.html")

@app.route('/submit', methods=['POST'])
def submit():
    form_data = {k: request.form[k] for k in [
        "age","sexe","etude","sommeil","distraction",
        "env","assiduite","ponctualite","discipline",
        "tache","niveau","moyenne"
    ]}

    student = process_student(form_data)
    # ── PRG pattern : on stocke en session puis on redirige ──
    session["last_student"] = vars(student)
    return redirect(url_for("resultat"))

# ─────────────────────────────────────────────────────────────────
# RÉSULTAT INDIVIDUEL  (GET uniquement → pas de re-POST)
# ─────────────────────────────────────────────────────────────────
@app.route('/resultat')
def resultat():
    student_dict = session.get("last_student")
    if not student_dict:
        return redirect(url_for("formulaire"))

    student = Student(**student_dict)
    data    = afficher_donnees()

    classe_finale = "indisponible"
    explication   = []

    if len(data) >= 10:
        data_cl = ajouter_classe(data.copy())
        if data_cl["classe"].nunique() >= 2:
            try:
                model_clf, scaler, _ = classification_modele(data_cl)
                features = ["etude","sommeil","distraction",
                            "assiduite","ponctualite","discipline","tache"]
                X_new     = pd.DataFrame(
                    [[float(student_dict[f]) for f in features]], columns=features)
                classe_ml = model_clf.predict(scaler.transform(X_new))[0]
                classe_finale = fusion_classe(float(student.moyenne), classe_ml)
                explication   = expliquer_classe(
                    float(student.moyenne), classe_ml, classe_finale)
            except Exception as e:
                print("Erreur classification :", e)

    moy_gen  = round(data["moyenne"].mean(), 2) if len(data) > 0 else "N/A"
    conseils = generer_conseils(student, classe_finale, moy_gen)

    return render_template("individuelle.html",
        student=student,
        classe_predite=classe_finale,
        moyenne_generale=moy_gen,
        conseils=conseils,
        explication=explication
    )

# ─────────────────────────────────────────────────────────────────
# ANALYSE GÉNÉRALE
# ─────────────────────────────────────────────────────────────────
@app.route('/generale')
def generale():
    data = afficher_donnees()
    if len(data) == 0:
        return render_template("generale.html", vide=True)

    data  = ajouter_classe(data)
    stats = stats_generales(data)

    # génération des graphiques dans la requête (pas de cache global)
    imgs = {
        "hist":    fig_b64(graphique_histogramme(data)),
        "classes": fig_b64(graphique_repartition_classes(data)),
        "sexe":    fig_b64(graphique_moyenne_sexe(data)),
        "niveau":  fig_b64(graphique_moyenne_niveau(data)),
        "corr":    fig_b64(graphique_correlations(data)),
        "boxplot": fig_b64(graphique_boxplot(data)),
    }

    # PCA (seulement si assez de données)
    imgs["pca"] = None
    variance    = None
    if len(data) >= 10:
        try:
            _, fig_pca, variance = analyser_pca(data)
            imgs["pca"] = fig_b64(fig_pca)
        except Exception as e:
            print("Erreur PCA :", e)

    return render_template("generale.html",
        vide=False, stats=stats,
        imgs=imgs, desc=DESCRIPTIONS,
        variance=variance
    )

# ─────────────────────────────────────────────────────────────────
# POINT DE SANTÉ (Render le ping pour garder le service actif)
# ─────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return {"status": "ok"}, 200

# ─────────────────────────────────────────────────────────────────
# LANCEMENT — NE PAS utiliser app.run() en production
# Gunicorn est configuré dans le Procfile
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 11000))
    app.run(host="0.0.0.0", port=port, debug=False)
