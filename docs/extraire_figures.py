"""Extrait des graphiques de résultats du notebook (sorties déjà exécutées) vers docs/ pour le README.

Les images sont copiées telles quelles depuis les sorties du notebook : aucun recalcul.

Usage (depuis la racine du projet) : python docs/extraire_figures.py
"""
import base64
import json
from pathlib import Path

DOSSIER_DOCS = Path(__file__).resolve().parent
NOTEBOOK = DOSSIER_DOCS.parent / "notebooks" / "01_modelisation.ipynb"

# Début du code de la cellule -> nom du fichier image (première image produite par la cellule)
FIGURES = {
    'fig, axes = plt.subplots(1, 2, figsize=(14, 5))': "fig_distribution_cible.png",
    'NOM_LISTE_FINALE = ': "fig_mae_jeux_features.png",
    'comparatif_modeles = pd.DataFrame(': "fig_mae_modeles.png",
    'erreur_signee_test = (prix_predit_test': "fig_test_reel_vs_predit.png",
    'noms_features = [nom.split(': "fig_importance_features.png",
}

cellules = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
for debut, nom in FIGURES.items():
    cellule = next(c for c in cellules if c["cell_type"] == "code" and "".join(c["source"]).startswith(debut))
    image = next(o["data"]["image/png"] for o in cellule["outputs"] if "image/png" in o.get("data", {}))
    chemin = DOSSIER_DOCS / nom
    chemin.write_bytes(base64.b64decode(image))
    print(f"{chemin.relative_to(DOSSIER_DOCS.parent)} ({chemin.stat().st_size / 1024:.0f} Ko)")
