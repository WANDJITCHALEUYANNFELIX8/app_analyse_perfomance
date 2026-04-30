import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from database import get_all_students

# ── palette ───────────────────────────────────────────────────────
NIGHT  = "#0D1B2A"
SOFT   = "#1E3A5F"
SKY    = "#4FC3F7"
GREEN  = "#00E5A0"
YELLOW = "#FFD600"
RED    = "#FF6B6B"
AMBER  = "#EF9F27"
WHITE  = "#E8F4FD"

# ── descriptions des graphiques ───────────────────────────────────
DESCRIPTIONS = {
    "histogramme": (
        "Distribution des moyennes",
        "Histogramme de toutes les notes de la promotion. "
        "Chaque barre représente un intervalle de 1 à 2 points. "
        "La couleur plus intense signale un plus grand nombre d'étudiants "
        "dans cet intervalle. La ligne jaune indique la médiane."
    ),
    "classes": (
        "Répartition par classe",
        "Diagramme en barres des 4 niveaux. "
        "Faible : < 10 | Moyen : 10–14 | Bon : 15–17 | Excellent : ≥ 18."
    ),
    "sexe": (
        "Moyenne par sexe",
        "Comparaison des performances moyennes entre étudiants masculins (M) "
        "et féminins (F)."
    ),
    "niveau": (
        "Moyenne par niveau d'études",
        "Évolution de la moyenne selon le niveau (L1 à M2). "
        "Permet de voir si les étudiants progressent au fil des années."
    ),
    "correlations": (
        "Corrélations avec la moyenne",
        "Coefficient de corrélation de Pearson entre chaque variable et la moyenne. "
        "Vert = influence positive. Rouge = influence négative."
    ),
    "boxplot": (
        "Boîtes à moustaches — variables comportementales",
        "Trait jaune = médiane. Boîte = Q1 à Q3 (50% des données). "
        "Moustaches = min/max. Points isolés = valeurs aberrantes."
    ),
    "pca": (
        "Projection PCA des étudiants",
        "Réduction des 7 variables comportementales en 2 dimensions. "
        "Chaque point = un étudiant. La couleur indique sa moyenne. "
        "Des points proches = profils similaires."
    ),
}

# ─────────────────────────────────────────────
# UTILITAIRE AXES
# ─────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
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
    return get_all_students()

# ─────────────────────────────────────────────
# ANALYSE DESCRIPTIVE
# ─────────────────────────────────────────────

def ajouter_classe(data):
    def definir_classe(m):
        if   m < 10: return "faible"
        elif m < 15: return "moyen"
        elif m < 18: return "bon"
        else:        return "excellent"
    data = data.copy()
    data["classe"] = data["moyenne"].apply(definir_classe)
    return data

def get_class_from_mean(m):
    if   m < 10: return "faible"
    elif m < 15: return "moyen"
    elif m < 18: return "bon"
    else:        return "excellent"

def stats_generales(data):
    m   = data["moyenne"]
    q1  = round(m.quantile(0.25), 2)
    q2  = round(m.quantile(0.50), 2)
    q3  = round(m.quantile(0.75), 2)
    iqr = round(q3 - q1, 2)
    variables = ["etude","sommeil","distraction",
                 "assiduite","ponctualite","discipline","tache"]
    return {
        "nb":          len(data),
        "moy_gen":     round(m.mean(),   2),
        "moy_mediane": q2,
        "moy_mode":    round(m.mode()[0], 2) if not m.mode().empty else "N/A",
        "moy_min":     round(m.min(),    2),
        "moy_max":     round(m.max(),    2),
        "moy_ecart":   round(m.std(),    2),
        "moy_var":     round(m.var(),    2),
        "q1":  q1, "q2": q2, "q3": q3, "iqr": iqr,
        "p10": round(m.quantile(0.10), 2),
        "p90": round(m.quantile(0.90), 2),
        "par_sexe":   data.groupby("sexe")["moyenne"].mean().round(2).to_dict(),
        "par_niveau": data.groupby("niveau")["moyenne"].mean().round(2).to_dict(),
        "nb_sexe":    data["sexe"].value_counts().to_dict(),
        "nb_niveau":  data["niveau"].value_counts().to_dict(),
        "corrs":      {v: round(data[v].corr(m), 3) for v in variables},
    }

# ─────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────

def graphique_histogramme(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    n, bins, patches = ax.hist(data["moyenne"], bins=12,
                                edgecolor=NIGHT, linewidth=0.8)
    max_n = max(n) if max(n) > 0 else 1
    for patch, val in zip(patches, n):
        t = val / max_n
        patch.set_facecolor(f"#{int(79+(0-79)*t):02x}{int(195+(229-195)*t):02x}{int(247+(160-247)*t):02x}")
    med = data["moyenne"].median()
    ax.axvline(med, color=YELLOW, linewidth=1.8, linestyle="--", label=f"Médiane : {med:.1f}")
    ax.legend(facecolor=SOFT, edgecolor="#2C4A6E", labelcolor=WHITE, fontsize=9)
    _style_ax(ax, title=DESCRIPTIONS["histogramme"][0],
              xlabel="Moyenne (/20)", ylabel="Nombre d'étudiants")
    plt.tight_layout()
    return fig

def graphique_repartition_classes(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    ordre = ["faible","moyen","bon","excellent"]
    rep   = data["classe"].value_counts().reindex(ordre, fill_value=0)
    bars  = ax.bar(rep.index, rep.values,
                   color=[RED,AMBER,SKY,GREEN], edgecolor=NIGHT, width=0.55)
    for bar, val in zip(bars, rep.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(val), ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["classes"][0],
              xlabel="Classe", ylabel="Nombre d'étudiants")
    plt.tight_layout()
    return fig

def graphique_moyenne_sexe(data):
    fig, ax = plt.subplots(figsize=(5, 4))
    moy  = data.groupby("sexe")["moyenne"].mean()
    bars = ax.bar(moy.index, moy.values,
                  color=[SKY,GREEN][:len(moy)], edgecolor=NIGHT, width=0.4)
    for bar, val in zip(bars, moy.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val:.2f}", ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["sexe"][0], xlabel="Sexe", ylabel="Moyenne (/20)")
    plt.tight_layout()
    return fig

def graphique_moyenne_niveau(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    moy  = data.groupby("niveau")["moyenne"].mean().sort_index()
    bars = ax.bar(moy.index, moy.values,
                  color=SKY, edgecolor=NIGHT, linewidth=0.8, width=0.5)
    for bar, val in zip(bars, moy.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val:.2f}", ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["niveau"][0], xlabel="Niveau", ylabel="Moyenne (/20)")
    plt.tight_layout()
    return fig

def graphique_correlations(data):
    variables = ["etude","sommeil","distraction",
                 "assiduite","ponctualite","discipline","tache"]
    corrs    = [round(data[v].corr(data["moyenne"]), 3) for v in variables]
    couleurs = [GREEN if c > 0 else RED for c in corrs]
    fig, ax  = plt.subplots(figsize=(7, 4))
    bars     = ax.barh(variables, corrs, color=couleurs, edgecolor=NIGHT)
    ax.axvline(0, color=WHITE, linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, corrs):
        ax.text(val+(0.01 if val>=0 else -0.01), bar.get_y()+bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if val>=0 else "right",
                color=WHITE, fontsize=9, fontweight="bold")
    _style_ax(ax, title=DESCRIPTIONS["correlations"][0],
              xlabel="Coefficient de corrélation")
    plt.tight_layout()
    return fig

def graphique_boxplot(data):
    variables = ["etude","sommeil","distraction",
                 "assiduite","ponctualite","discipline","tache"]
    fig, ax   = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot([data[v].dropna() for v in variables], labels=variables,
                    patch_artist=True,
                    medianprops=dict(color=YELLOW, linewidth=2),
                    whiskerprops=dict(color=WHITE),
                    capprops=dict(color=WHITE),
                    flierprops=dict(marker="o", color=RED, markersize=4, alpha=0.6))
    for patch, color in zip(bp["boxes"], [SKY,GREEN,RED,SKY,GREEN,SKY,GREEN]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_xticklabels(variables, rotation=25, ha="right", color=WHITE, fontsize=9)
    _style_ax(ax, title=DESCRIPTIONS["boxplot"][0], ylabel="Valeur")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────

def analyser_pca(data):
    features = ["etude","sommeil","distraction",
                "assiduite","ponctualite","discipline","tache"]
    X        = data[features].dropna()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca      = PCA(n_components=2)
    comps    = pca.fit_transform(X_scaled)
    variance = pca.explained_variance_ratio_

    df_pca = pd.DataFrame({
        "PC1":     comps[:, 0],
        "PC2":     comps[:, 1],
        "moyenne": data["moyenne"].values[:len(comps)],
        "classe":  data["classe"].values[:len(comps)]
    })

    # graphique
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df_pca["PC1"], df_pca["PC2"],
                    c=df_pca["moyenne"], cmap="viridis",
                    s=80, alpha=0.9, edgecolors=NIGHT)
    cb = plt.colorbar(sc, ax=ax, label="Moyenne")
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)
    cb.set_label("Moyenne", color=WHITE, fontsize=9)
    pct1 = round(variance[0]*100, 1)
    pct2 = round(variance[1]*100, 1)
    _style_ax(ax,
              title=DESCRIPTIONS["pca"][0],
              xlabel=f"PC1 ({pct1}% de variance)",
              ylabel=f"PC2 ({pct2}% de variance)")
    plt.tight_layout()

    return df_pca, fig, variance

# ─────────────────────────────────────────────
# CLASSIFICATION
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

def fusion_classe(moyenne, classe_ml):
    """Combine règle métier (70%) + ML (30%)."""
    poids = {"faible":1, "moyen":2, "bon":3, "excellent":4}
    score = 0.7 * poids[get_class_from_mean(moyenne)] + 0.3 * poids.get(classe_ml, 2)
    if   score < 1.5: return "faible"
    elif score < 2.5: return "moyen"
    elif score < 3.5: return "bon"
    else:             return "excellent"

def expliquer_classe(moyenne, classe_ml, classe_finale):
    exp = []
    seuils = [(10,"Ta moyenne de {m} montre des difficultés dans plusieurs matières."),
              (15,"Ta moyenne de {m} est correcte, mais tu peux progresser."),
              (18,"Ta moyenne de {m} est bonne, continue ainsi."),
              (21,"Ta moyenne de {m} est excellente, tu es parmi les meilleurs.")]
    for seuil, msg in seuils:
        if moyenne < seuil:
            exp.append("📊 " + msg.format(m=moyenne)); break
    labels_ml = {"faible":"🤖 Ton comportement indique des efforts encore insuffisants.",
                 "moyen": "🤖 Ton comportement est moyen : régulier, mais pas encore optimal.",
                 "bon":   "🤖 Ton comportement est bon : tu es discipliné et sérieux.",
                 "excellent":"🤖 Ton comportement est excellent : très organisé et régulier."}
    exp.append(labels_ml.get(classe_ml, "🤖 Analyse comportementale effectuée."))
    labels_fin = {"faible":"⚠️ Résultat : niveau faible. Change ta méthode de travail.",
                  "moyen": "📌 Résultat : niveau moyen. Sois plus constant.",
                  "bon":   "👍 Résultat : bon niveau. Continue tes efforts.",
                  "excellent":"🌟 Résultat : excellent niveau. Félicitations !"}
    exp.append(labels_fin.get(classe_finale, ""))
    exp.append("💡 La régularité et la discipline sont les clés de ta réussite.")
    return exp

def generer_conseils(student, classe_predite, moyenne_generale):
    conseils = []
    c = str(classe_predite).lower()

    if float(student.etude) < 4:
        conseils.append("📚 Augmente ton temps d'étude — vise au moins 4 h/jour.")
    elif float(student.etude) > 8:
        conseils.append("⚠️ Attention au surmenage. Un excès d'étude peut réduire l'efficacité.")
    if float(student.sommeil) < 6:
        conseils.append("💤 Dors au moins 7 à 8 h — essentiel pour la mémoire et la concentration.")
    elif float(student.sommeil) > 9:
        conseils.append("⚠️ Un excès de sommeil peut aussi affecter ta productivité.")
    if float(student.distraction) > 6:
        conseils.append("📵 Réduis les distractions pendant les sessions de travail.")
    if float(student.assiduite) < 6:
        conseils.append("🏫 Améliore ton assiduité — être présent en cours est essentiel.")
    if float(student.ponctualite) < 6:
        conseils.append("⏰ La ponctualité te permet de ne pas manquer les débuts de cours.")
    if float(student.discipline) < 6:
        conseils.append("🎯 Planifie tes révisions avec un calendrier fixe.")
    if float(student.tache) < 6:
        conseils.append("✅ Fais tes tâches régulièrement — ne reporte pas.")
    if float(student.env) < 3:
        conseils.append("🏠 Améliore ton environnement : calme, éclairé, organisé.")

    if not conseils:
        messages = {"excellent":"🌟 Continue sur cette lancée !",
                    "bon":      "👍 Quelques ajustements peuvent te faire passer excellent.",
                    "moyen":    "📈 Tu peux progresser avec plus de discipline.",
                    "faible":   "⚠️ Un changement sérieux de méthode est recommandé."}
        conseils.append(messages.get(c, "💪 Reste motivé·e et régulier·e."))

    if moyenne_generale != "N/A":
        diff = float(student.moyenne) - float(moyenne_generale)
        if diff < 0:
            conseils.append(f"📊 Ta moyenne est en dessous de la promo ({moyenne_generale}). Un effort supplémentaire est nécessaire.")
        else:
            conseils.append(f"📊 Ta moyenne est au-dessus de la promo ({moyenne_generale}). Continue ainsi !")

    priorite = None
    if float(student.distraction) > 3:
        priorite = "Réduire les distractions"
    elif float(student.etude) < 4:
        priorite = "Augmenter le temps d'étude"
    elif float(student.sommeil) < 4:
        priorite = "Améliorer le sommeil"
    if priorite:
        conseils.insert(0, f"🔥 PRIORITÉ : {priorite}")

    return conseils
