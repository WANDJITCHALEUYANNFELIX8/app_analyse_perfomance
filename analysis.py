import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sqlite3

DB_NAME = "student.db"

# ── palette ───────────────────────────────────────────────────────
NIGHT  = "#0D1B2A"
SOFT   = "#1E3A5F"
SKY    = "#4FC3F7"
GREEN  = "#00E5A0"
YELLOW = "#FFD600"
RED    = "#FF6B6B"
AMBER  = "#EF9F27"
WHITE  = "#E8F4FD"

# ── descriptions des graphiques (utilisées dans les templates) ────
DESCRIPTIONS = {
    "histogramme": (
        "Distribution des moyennes",
        "Histogramme de toutes les notes de la promotion. "
        "Chaque barre représente un intervalle de 1 à 2 points. "
        "La couleur plus intense signale un plus grand nombre d'étudiants "
        "dans cet intervalle. Un pic à droite indique de bons résultats généraux."
    ),
    "nuage": (
        "Étude vs Moyenne — Nuage de points",
        "Chaque point représente un étudiant. L'axe horizontal = temps d'étude "
        "quotidien (h/j), l'axe vertical = moyenne obtenue (/20). "
        "La couleur varie du bleu (basses notes) au vert (hautes notes). "
        "Une tendance montante confirme l'impact positif du travail sur les résultats."
    ),
    "classes": (
        "Répartition par classe",
        "Diagramme en barres des 4 niveaux académiques. "
        "Faible : moyenne < 10 | Moyen : 10–14 | Bon : 15–17 | Excellent : ≥ 18. "
        "La hauteur de chaque barre indique le nombre d'étudiants dans cette classe."
    ),
    "sexe": (
        "Moyenne par sexe",
        "Comparaison des performances moyennes entre étudiants masculins (M) "
        "et féminins (F). Permet de détecter d'éventuelles différences de résultats "
        "selon le genre dans la promotion."
    ),
    "niveau": (
        "Moyenne par niveau d'études",
        "Évolution de la moyenne selon le niveau (L1 à M2). "
        "Permet de voir si les étudiants progressent au fil des années "
        "ou si certains niveaux présentent des difficultés particulières."
    ),
    "correlations": (
        "Corrélations avec la moyenne",
        "Barres horizontales montrant le coefficient de corrélation de Pearson "
        "entre chaque variable et la moyenne. "
        "Vert = influence positive (plus la valeur est haute, meilleure est la note). "
        "Rouge = influence négative (ex : distraction élevée → mauvaise note). "
        "Une valeur proche de ±1 indique une forte relation."
    ),
    "boxplot": (
        "Boîtes à moustaches — variables comportementales",
        "Chaque boîte montre la distribution d'une variable : "
        "le trait central = médiane, la boîte = 1er au 3e quartile (50% des données), "
        "les moustaches = valeurs min/max hors valeurs aberrantes (points isolés)."
    ),
}

def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Applique le thème sombre à un axe matplotlib."""
    ax.set_facecolor(SOFT)
    ax.figure.set_facecolor(NIGHT)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C4A6E")
    ax.grid(True, color="#2C4A6E", linewidth=0.6, linestyle="--", alpha=0.7)
    if title:  ax.set_title(title,  fontsize=12, fontweight="bold", pad=12, color=WHITE)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, labelpad=8)

# ─────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────

def afficher_donnees():
    conn = sqlite3.connect(DB_NAME)
    df   = pd.read_sql_query("SELECT * FROM student", conn)
    conn.close()
    return df

# ─────────────────────────────────────────────
# ANALYSE DESCRIPTIVE
# ─────────────────────────────────────────────

def stats_generales(data):
    """
    Retourne un dictionnaire complet de statistiques descriptives :
    tendance centrale, dispersion, quantiles, répartition par groupe,
    corrélations.
    """
    m = data["moyenne"]

    # ── quantiles ──
    q1  = round(m.quantile(0.25), 2)
    q2  = round(m.quantile(0.50), 2)   # = médiane
    q3  = round(m.quantile(0.75), 2)
    iqr = round(q3 - q1, 2)            # écart interquartile

    # ── corrélations ──
    variables = ["etude","sommeil","distraction",
                 "assiduite","ponctualite","discipline","tache"]
    corrs = {v: round(data[v].corr(m), 3) for v in variables}

    return {
        # tendance centrale
        "nb":          len(data),
        "moy_gen":     round(m.mean(),   2),
        "moy_mediane": q2,
        "moy_mode":    round(m.mode()[0], 2) if not m.mode().empty else "N/A",
        # dispersion
        "moy_min":     round(m.min(),    2),
        "moy_max":     round(m.max(),    2),
        "moy_ecart":   round(m.std(),    2),
        "moy_var":     round(m.var(),    2),
        # quantiles
        "q1":          q1,
        "q2":          q2,
        "q3":          q3,
        "iqr":         iqr,
        "p10":         round(m.quantile(0.10), 2),
        "p90":         round(m.quantile(0.90), 2),
        # répartitions
        "par_sexe":    data.groupby("sexe")["moyenne"].mean().round(2).to_dict(),
        "par_niveau":  data.groupby("niveau")["moyenne"].mean().round(2).to_dict(),
        "nb_sexe":     data["sexe"].value_counts().to_dict(),
        "nb_niveau":   data["niveau"].value_counts().to_dict(),
        # corrélations
        "corrs":       corrs,
    }

def ajouter_classe(data):
    def definir_classe(m):
        if   m < 10: return "faible"
        elif m < 15: return "moyen"
        elif m < 18: return "bon"
        else:        return "excellent"
    data = data.copy()
    data["classe"] = data["moyenne"].apply(definir_classe)
    return data

# ─────────────────────────────────────────────
# GRAPHIQUES DESCRIPTIFS
# ─────────────────────────────────────────────

def graphique_histogramme(data):
    """Distribution des moyennes avec dégradé de couleur."""
    fig, ax = plt.subplots(figsize=(7, 4))
    n, bins, patches = ax.hist(data["moyenne"], bins=12,
                                edgecolor=NIGHT, linewidth=0.8)
    max_n = max(n) if max(n) > 0 else 1
    for patch, val in zip(patches, n):
        t = val / max_n
        r = int(79  + (0   - 79)  * t)
        g = int(195 + (229 - 195) * t)
        b = int(247 + (160 - 247) * t)
        patch.set_facecolor(f"#{r:02x}{g:02x}{b:02x}")
    # ligne médiane
    med = data["moyenne"].median()
    ax.axvline(med, color=YELLOW, linewidth=1.8, linestyle="--", label=f"Médiane : {med:.1f}")
    ax.legend(facecolor=SOFT, edgecolor="#2C4A6E", labelcolor=WHITE, fontsize=9)
    _style_ax(ax, title=DESCRIPTIONS["histogramme"][0],
              xlabel="Moyenne (/20)", ylabel="Nombre d'étudiants")
    plt.tight_layout()
    return fig

def graphique_nuage_etude(data):
    """Nuage de points étude vs moyenne."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(data["etude"], data["moyenne"],
                    c=data["moyenne"], cmap="cool",
                    s=70, alpha=0.85, edgecolors=NIGHT, linewidths=0.5)
    cb = plt.colorbar(sc, ax=ax)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)
    cb.set_label("Moyenne", color=WHITE, fontsize=9)
    _style_ax(ax, title=DESCRIPTIONS["nuage"][0],
              xlabel="Temps d'étude (h/jour)", ylabel="Moyenne (/20)")
    plt.tight_layout()
    return fig

def graphique_repartition_classes(data):
    """Répartition des classes avec labels sur les barres."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ordre    = ["faible", "moyen", "bon", "excellent"]
    couleurs = [RED, AMBER, SKY, GREEN]
    repartition = data["classe"].value_counts().reindex(ordre, fill_value=0)
    bars = ax.bar(repartition.index, repartition.values,
                  color=couleurs, edgecolor=NIGHT, linewidth=0.8, width=0.55)
    for bar, val in zip(bars, repartition.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom",
                color=WHITE, fontsize=10, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["classes"][0],
              xlabel="Classe", ylabel="Nombre d'étudiants")
    plt.tight_layout()
    return fig

def graphique_moyenne_sexe(data):
    """Moyenne par sexe."""
    fig, ax = plt.subplots(figsize=(5, 4))
    moy = data.groupby("sexe")["moyenne"].mean()
    bars = ax.bar(moy.index, moy.values,
                  color=[SKY, GREEN][:len(moy)], edgecolor=NIGHT, width=0.4)
    for bar, val in zip(bars, moy.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom",
                color=WHITE, fontsize=10, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["sexe"][0],
              xlabel="Sexe", ylabel="Moyenne (/20)")
    plt.tight_layout()
    return fig

def graphique_moyenne_niveau(data):
    """Moyenne par niveau d'études."""
    fig, ax = plt.subplots(figsize=(7, 4))
    moy = data.groupby("niveau")["moyenne"].mean().sort_index()
    bars = ax.bar(moy.index, moy.values,
                  color=SKY, edgecolor=NIGHT, linewidth=0.8, width=0.5)
    for bar, val in zip(bars, moy.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom",
                color=WHITE, fontsize=10, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["niveau"][0],
              xlabel="Niveau", ylabel="Moyenne (/20)")
    plt.tight_layout()
    return fig

def graphique_correlations(data):
    """Corrélations de toutes les variables avec la moyenne."""
    variables = ["etude","sommeil","distraction",
                 "assiduite","ponctualite","discipline","tache"]
    corrs    = [round(data[v].corr(data["moyenne"]), 3) for v in variables]
    couleurs = [GREEN if c > 0 else RED for c in corrs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(variables, corrs, color=couleurs,
                   edgecolor=NIGHT, linewidth=0.8)
    ax.axvline(0, color=WHITE, linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, corrs):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                color=WHITE, fontsize=9, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["correlations"][0],
              xlabel="Coefficient de corrélation", ylabel="")
    plt.tight_layout()
    return fig

def graphique_boxplot(data):
    """Boîtes à moustaches des variables comportementales."""
    variables = ["etude","sommeil","distraction",
                 "assiduite","ponctualite","discipline","tache"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(
        [data[v].dropna() for v in variables],
        labels=variables,
        patch_artist=True,
        medianprops=dict(color=YELLOW, linewidth=2),
        whiskerprops=dict(color=WHITE),
        capprops=dict(color=WHITE),
        flierprops=dict(marker="o", color=RED, markersize=4, alpha=0.6)
    )
    colors = [SKY, GREEN, RED, SKY, GREEN, SKY, GREEN]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_xticklabels(variables, rotation=25, ha="right", color=WHITE, fontsize=9)
    _style_ax(ax, title=DESCRIPTIONS["boxplot"][0], ylabel="Valeur")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# CLASSIFICATION (résultat individuel)
# ─────────────────────────────────────────────

def classification_modele(data):
    features = ["etude","sommeil","distraction","assiduite",
                "ponctualite","discipline","tache"]
    X        = data[features]
    y        = data["classe"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, acc

def predire_classe(model, scaler, form_data):
    """Prédit la classe d'un nouvel étudiant."""
    features = ["etude","sommeil","distraction","assiduite",
                "ponctualite","discipline","tache"]
    X_new = pd.DataFrame(
        [[float(form_data[f]) for f in features]], columns=features)
    return model.predict(scaler.transform(X_new))[0]

def generer_conseils(student, classe_predite,moyenne_generale):
    """Génère des conseils personnalisés selon le profil de l'étudiant."""
    conseils = []
    c = str(classe_predite).lower()

    if float(student.etude) < 4:
        conseils.append("📚 Augmente ton temps d'étude quotidien — vise au moins 4 h/jour.")
    elif float(student.etude) > 8:
        conseils.append("⚠️ Attention au surmenage. Un excès d'étude peut réduire l'efficacité.")    
        
    if float(student.sommeil) < 6:
        conseils.append("💤 Tu manques de sommeil. 7 à 8 h par nuit améliorent la mémoire et la concentration.")
    elif float(student.sommeil) > 9:
        conseils.append("Un excès de sommeil peut aussi affecter votre productivité.")    
        
    if float(student.distraction) > 6:
        conseils.append("📵 Réduis les distractions (téléphone, réseaux sociaux) pendant les sessions de travail.")
    if float(student.assiduite) < 6:
        conseils.append("🏫 Améliore ton assiduité — être présent en cours est la base de la réussite.")
    if float(student.ponctualite) < 6:
        conseils.append("⏰ La ponctualité permet de ne pas manquer les débuts de cours, souvent essentiels.")
    if float(student.discipline) < 6:
        conseils.append("🎯 Renforce ta discipline : planifie tes révisions avec un calendrier fixe.")
    if float(student.tache) < 6:
        conseils.append("✅ Fais tes tâches et devoirs régulièrement — ne reporte pas à la dernière minute.")
    if float(student.env) < 3:
        conseils.append("🏠 Améliore ton environnement de travail : espace calme, bien éclairé, organisé.")

    if not conseils:
        if "excellent" in c:
            conseils.append("🌟 Profil excellent ! Continue sur cette lancée et partage tes méthodes avec tes camarades.")
        elif "bon" in c:
            conseils.append("👍 Très bon profil. Quelques ajustements mineurs peuvent te faire basculer en excellent.")
        elif "faible"in c:
            conseils.append("⚠️ Votre niveau est bas . Un changement sérieux de méthode de travail est recommandé.")
        elif classe_predite == "moyen":
            conseils.append("📈 Vous avez un niveau moyen. Vous pouvez facilement progresser avec plus de discipline.")    
        else:
            conseils.append("📈 Vous avez un niveau moyen. Vous pouvez facilement progresser avec plus de discipline.\n💪 Reste motivé·e et régulier·e. La constance est la clé de la progression.")
            
            
     
    # ─────────────────────────────
    # 3. COMPARAISON AVEC LA MOYENNE GLOBALE
    # ─────────────────────────────
    if moyenne_generale != "N/A":
        if student.moyenne < moyenne_generale:
            conseils.append("Votre moyenne est en dessous de la moyenne générale. Un effort supplémentaire est nécessaire.")
        else:
            conseils.append("Votre moyenne est au-dessus de la moyenne générale. Continuez ainsi !")       
            
    priorite = None

    if student.distraction > 3:
        priorite = "Réduire les distractions"
    elif student.etude < 4:
        priorite = "Augmenter le temps d'étude"
    elif student.sommeil < 4:
        priorite = "Améliorer le sommeil"

    if priorite:
        conseils.insert(0, f"🔥 PRIORITÉ : {priorite}")        

    return conseils
