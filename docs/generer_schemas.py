"""Génère les 3 schémas pédagogiques de la veille (README) dans le dossier docs/.

- schema_regression_lineaire.png : points, droite ajustée et résidus
- schema_random_forest.png      : plusieurs arbres indépendants, puis moyenne
- schema_gradient_boosting.png  : arbres successifs qui corrigent les erreurs

Les données affichées sont FICTIVES (générées aléatoirement) : elles servent
uniquement à illustrer le fonctionnement des algorithmes.

Usage (depuis la racine du projet) : python docs/generer_schemas.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans fenêtre, en fichier uniquement
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import GradientBoostingRegressor

DOSSIER_DOCS = Path(__file__).resolve().parent
RANDOM_STATE = 42

# Palette sobre, cohérente avec les graphiques du notebook
BLEU = "steelblue"
BLEU_CLAIR = "#dce8f4"
GRIS = "#b0b7c3"
GRIS_TEXTE = "#5b6675"
ROUGE = "#c0392b"
VERT = "seagreen"
ENCRE = "#1c2430"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def boite(ax, x, y, largeur, hauteur, texte, couleur_fond=BLEU_CLAIR, couleur_bord=BLEU, taille=10, gras=False):
    """Boîte arrondie centrée en (x, y) avec un texte."""
    ax.add_patch(FancyBboxPatch((x - largeur / 2, y - hauteur / 2), largeur, hauteur,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=couleur_fond, edgecolor=couleur_bord, linewidth=1.5))
    ax.text(x, y, texte, ha="center", va="center", fontsize=taille, color=ENCRE,
            fontweight="bold" if gras else "normal")


def fleche(ax, depart, arrivee, couleur=GRIS_TEXTE):
    ax.add_patch(FancyArrowPatch(depart, arrivee, arrowstyle="-|>", mutation_scale=14,
                                 color=couleur, linewidth=1.4, shrinkA=2, shrinkB=2))


def mini_arbre(ax, x, y, echelle=1.0, couleur=BLEU, profondeur=3, graine=0):
    """Dessine un petit arbre de décision binaire (nœuds + branches) dont la racine est en (x, y)."""
    rng = np.random.default_rng(graine)
    niveaux = [[(x, y)]]
    for niveau in range(1, profondeur):
        ecart = 0.55 * echelle / (2 ** (niveau - 1))
        enfants = []
        for (px, py) in niveaux[-1]:
            # Quelques nœuds restent des feuilles : les arbres n'ont pas tous la même forme
            if niveau > 1 and rng.random() < 0.3:
                continue
            for signe in (-1, 1):
                cx, cy = px + signe * ecart, py - 0.42 * echelle
                ax.plot([px, cx], [py, cy], color=GRIS_TEXTE, linewidth=1.1, zorder=1)
                enfants.append((cx, cy))
        niveaux.append(enfants)
    for niveau in niveaux:
        for (nx, ny) in niveau:
            ax.add_patch(plt.Circle((nx, ny), 0.075 * echelle, color=couleur, zorder=2))


# ---------------------------------------------------------------------------
# 1. Régression linéaire : points, droite des moindres carrés, résidus
# ---------------------------------------------------------------------------
def schema_regression_lineaire():
    rng = np.random.default_rng(RANDOM_STATE)
    surface = rng.uniform(60, 220, 18)
    prix = 1.6 * surface + 40 + rng.normal(0, 28, surface.size)

    pente, ordonnee = np.polyfit(surface, prix, deg=1)
    prix_predit = pente * surface + ordonnee

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for xi, yi, pi in zip(surface, prix, prix_predit):
        ax.plot([xi, xi], [yi, pi], color=ROUGE, linewidth=1.3, alpha=0.8, zorder=1)
    ax.scatter(surface, prix, s=46, color=BLEU, zorder=3, label="Maisons observées")
    grille = np.linspace(50, 230, 2)
    ax.plot(grille, pente * grille + ordonnee, color=ENCRE, linewidth=2.2, zorder=2,
            label=f"Droite ajustée : prix = {pente:.2f} × surface + {ordonnee:.0f}".replace(".", ","))
    ax.plot([], [], color=ROUGE, linewidth=1.3, label="Résidus (erreurs) : réel − prédit")

    # Annotation d'un résidu
    i = int(np.argmax(np.abs(prix - prix_predit)))
    ax.annotate("résidu", xy=(surface[i], (prix[i] + prix_predit[i]) / 2), xytext=(surface[i] + 18, (prix[i] + prix_predit[i]) / 2 + 45),
                fontsize=10, color=ROUGE, arrowprops={"arrowstyle": "->", "color": ROUGE})

    ax.set_xlabel("Surface habitable (m²)")
    ax.set_ylabel("Prix (milliers de $)")
    ax.set_title("Régression linéaire : la droite qui minimise la somme des résidus²", fontsize=13, pad=12)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.99, 0.02, "Données fictives, à visée pédagogique", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=GRIS_TEXTE)
    fig.tight_layout()
    fig.savefig(DOSSIER_DOCS / "schema_regression_lineaire.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Random Forest : échantillons bootstrap -> arbres indépendants -> moyenne
# ---------------------------------------------------------------------------
def schema_random_forest():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    boite(ax, 5.5, 5.75, 3.6, 0.6, "Données d'entraînement", gras=True)

    positions = [1.6, 4.2, 6.8, 9.4]
    predictions = ["612 k$", "570 k$", "598 k$", "588 k$"]
    for k, (x, pred) in enumerate(zip(positions, predictions)):
        etiquette = f"Arbre {k + 1}" if k < 3 else "Arbre n"
        fleche(ax, (5.5, 5.42), (x, 4.95))
        boite(ax, x, 4.7, 2.1, 0.48, "échantillon bootstrap\n+ features tirées au hasard", taille=8.5,
              couleur_fond="white", couleur_bord=GRIS)
        mini_arbre(ax, x, 4.05, echelle=1.0, profondeur=4, graine=k + 3)
        ax.text(x, 2.45, etiquette, ha="center", fontsize=10.5, fontweight="bold", color=ENCRE)
        fleche(ax, (x, 2.3), (x, 1.95))
        boite(ax, x, 1.7, 1.5, 0.45, pred, couleur_fond="white", couleur_bord=BLEU, taille=10.5)
        fleche(ax, (x, 1.45), (5.5, 0.85))
    ax.text(8.1, 2.45, "…", ha="center", fontsize=16, color=GRIS_TEXTE)

    boite(ax, 5.5, 0.55, 4.4, 0.6, "Prédiction finale = moyenne des arbres ≈ 592 k$",
          couleur_fond=BLEU, couleur_bord=BLEU, taille=11, gras=True)
    ax.texts[-1].set_color("white")

    ax.set_title("Random Forest : des arbres entraînés en parallèle, indépendamment, puis moyennés",
                 fontsize=13, pad=4)
    ax.text(10.95, 0.02, "Valeurs fictives, à visée pédagogique", ha="right", fontsize=9, color=GRIS_TEXTE)
    fig.tight_layout()
    fig.savefig(DOSSIER_DOCS / "schema_random_forest.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Gradient Boosting : arbres successifs entraînés sur les erreurs résiduelles
# ---------------------------------------------------------------------------
def schema_gradient_boosting():
    fig = plt.figure(figsize=(12, 7.6))
    grille = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.35, wspace=0.25)

    # Haut : enchaînement des arbres
    ax = fig.add_subplot(grille[0, :])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    etapes = [
        (1.0, "Départ", "prédiction =\nmoyenne des prix"),
        (3.6, "Arbre 1", "apprend les\nerreurs restantes"),
        (6.2, "Arbre 2", "apprend les erreurs\nque l'arbre 1 a laissées"),
        (8.8, "Arbre 3", "corrige encore\nun peu…"),
    ]
    for k, (x, titre, texte) in enumerate(etapes):
        if k == 0:
            boite(ax, x, 1.75, 1.7, 1.2, f"{titre}\n\n{texte}", couleur_fond="white", couleur_bord=GRIS, taille=9.5)
        else:
            mini_arbre(ax, x, 2.55, echelle=0.8, profondeur=3, graine=k)
            ax.text(x, 1.45, titre, ha="center", fontsize=10.5, fontweight="bold", color=ENCRE)
            ax.text(x, 0.95, texte, ha="center", va="top", fontsize=9, color=GRIS_TEXTE)
        if k < len(etapes) - 1:
            fleche(ax, (x + 0.95, 1.75), (etapes[k + 1][0] - 0.95, 1.75), couleur=ROUGE)
            ax.text((x + etapes[k + 1][0]) / 2, 1.95, "résidus", ha="center", fontsize=9, color=ROUGE)
    fleche(ax, (9.75, 1.75), (10.35, 1.75))
    boite(ax, 11.2, 1.75, 1.5, 1.2, "Prédiction =\ndépart\n+ η·arbre 1\n+ η·arbre 2\n+ …",
          couleur_fond=BLEU_CLAIR, couleur_bord=BLEU, taille=9.5)
    ax.text(6, 0.05, "η = learning_rate : chaque arbre n'apporte qu'une fraction de sa correction",
            ha="center", fontsize=9.5, color=GRIS_TEXTE, style="italic")
    ax.set_title("Gradient Boosting : des arbres construits les uns après les autres, chacun corrige les erreurs des précédents",
                 fontsize=13, pad=6)

    # Bas : ajustement réel d'un GradientBoostingRegressor sur des données 1D fictives
    rng = np.random.default_rng(RANDOM_STATE)
    x = np.sort(rng.uniform(0, 10, 120))
    y = np.sin(x) * 2 + 0.3 * x + rng.normal(0, 0.35, x.size)
    X = x.reshape(-1, 1)
    modele = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=RANDOM_STATE).fit(X, y)
    x_fin = np.linspace(0, 10, 400).reshape(-1, 1)
    predictions_par_etape = list(modele.staged_predict(x_fin))

    for colonne, n_arbres in enumerate([1, 10, 100]):
        axe = fig.add_subplot(grille[1, colonne])
        axe.scatter(x, y, s=12, color=GRIS, label="données")
        axe.plot(x_fin, predictions_par_etape[n_arbres - 1], color=BLEU, linewidth=2.2, label="prédiction")
        axe.set_title(f"Après {n_arbres} arbre{'s' if n_arbres > 1 else ''}", fontsize=11)
        axe.set_xticks([])
        axe.set_yticks([])
        if colonne == 0:
            axe.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.99, 0.01, "Données fictives : ajustement réel d'un GradientBoostingRegressor (profondeur 2, η = 0,1)",
             ha="right", fontsize=9, color=GRIS_TEXTE)
    fig.savefig(DOSSIER_DOCS / "schema_gradient_boosting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    schema_regression_lineaire()
    schema_random_forest()
    schema_gradient_boosting()
    for nom in ["schema_regression_lineaire.png", "schema_random_forest.png", "schema_gradient_boosting.png"]:
        chemin = DOSSIER_DOCS / nom
        print(f"{chemin.relative_to(DOSSIER_DOCS.parent)} ({chemin.stat().st_size / 1024:.0f} Ko)")
