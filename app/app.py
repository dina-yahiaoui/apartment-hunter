"""Apartment Hunter : application Flask d'estimation du prix d'une maison (comté de King, 2014-2015)."""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# --- Chemins construits à partir de l'emplacement de ce fichier (fonctionne quel que soit le dossier courant) ---
DOSSIER_APP = Path(__file__).resolve().parent
DOSSIER_MODELES = DOSSIER_APP.parent / "models"

# --- Chargement unique au démarrage ---
MODELE = joblib.load(DOSSIER_MODELES / "model.joblib")
METADONNEES = json.loads((DOSSIER_MODELES / "metadata.json").read_text(encoding="utf-8"))
ZIPCODE_LATLONG = (
    pd.read_csv(DOSSIER_MODELES / "zipcode_latlong.csv")
    .set_index("zipcode")[["lat", "long"]]
    .to_dict(orient="index")
)

FEATURES = METADONNEES["features"]
ZIPCODES = METADONNEES["zipcodes"]
ANNEE_REFERENCE = METADONNEES["annee_reference_prix"]
ERREUR_TYPIQUE = METADONNEES["metriques_test"]["erreur relative médiane"]

# --- Libellés des échelles du comté de King ---
LIBELLES_GRADE = {
    1: "1 – Construction très sommaire",
    2: "2 – Construction très sommaire",
    3: "3 – Médiocre",
    4: "4 – Ancienne, de faible qualité",
    5: "5 – Passable",
    6: "6 – Modeste",
    7: "7 – Standard",
    8: "8 – Bonne",
    9: "9 – Très bonne",
    10: "10 – Supérieure",
    11: "11 – Excellente",
    12: "12 – Luxe",
    13: "13 – Prestige (demeure d'architecte)",
}
LIBELLES_VUE = {
    0: "Aucune vue particulière",
    1: "Vue correcte",
    2: "Vue moyenne",
    3: "Belle vue",
    4: "Vue exceptionnelle",
}
LIBELLES_ETAT = {
    1: "1 – Mauvais état (gros travaux)",
    2: "2 – État passable",
    3: "3 – État moyen, correct",
    4: "4 – Bon état",
    5: "5 – Excellent état",
}

# --- Bornes de validation (couvrent les valeurs observées dans les données d'entraînement) ---
BORNES = {
    "sqft_living": (250, 15_000),
    "sqft_lot": (500, 2_000_000),
    "sqft_basement": (0, 5_000),
    "annee_construction": (1900, ANNEE_REFERENCE),
}
BORNES_LAT = (47.15, 47.80)
BORNES_LONG = (-122.55, -121.30)

# Maison exemple du notebook (section 6.6) : valeurs pré-remplies du formulaire
VALEURS_PAR_DEFAUT = {
    "sqft_living": "1800",
    "sqft_lot": "4000",
    "sqft_basement": "600",
    "grade": "7",
    "condition": "3",
    "view": "0",
    "waterfront": "",
    "renovated": "",
    "annee_construction": "1950",
    "zipcode": "98103",
}

app = Flask(__name__)


def formater_nombre(nombre):
    """2000000 -> '2 000 000' (espace fine insécable) ; les nombres à 4 chiffres comme les années restent intacts."""
    return f"{nombre:,.0f}".replace(",", " ") if abs(nombre) >= 10_000 else f"{nombre:.0f}"


def formater_dollars(montant):
    """581568.3 -> '581 568 $'."""
    return formater_nombre(montant) + " $"


def lire_entier(formulaire, champ, libelle, erreurs, bornes=None, choix=None):
    """Lit un entier du formulaire et vérifie ses bornes ; enregistre un message d'erreur sinon."""
    # Espaces tolérés comme séparateurs de milliers (« 1 800 »)
    brut = re.sub(r"\s", "", formulaire.get(champ, ""))
    if brut == "":
        erreurs[champ] = f"{libelle} : ce champ est obligatoire."
        return None
    if not re.fullmatch(r"-?\d+", brut):
        erreurs[champ] = f"{libelle} : saisissez un nombre entier, sans décimale."
        return None
    valeur = int(brut)
    if bornes and not bornes[0] <= valeur <= bornes[1]:
        erreurs[champ] = f"{libelle} : la valeur doit être comprise entre {formater_nombre(bornes[0])} et {formater_nombre(bornes[1])}."
        return None
    if choix is not None and valeur not in choix:
        erreurs[champ] = f"{libelle} : choisissez une valeur dans la liste."
        return None
    return valeur


def lire_coordonnee(formulaire, champ, bornes):
    """Lit lat ou long ; renvoie None si absente, lève ValueError si invalide ou hors du comté."""
    brut = formulaire.get(champ, "").strip()
    if brut == "":
        return None
    valeur = float(brut)
    if not bornes[0] <= valeur <= bornes[1]:
        raise ValueError
    return valeur


def valider_formulaire(formulaire):
    """Valide la saisie et construit la maison (12 features). Renvoie (maison, erreurs)."""
    erreurs = {}
    maison = {
        "sqft_living": lire_entier(formulaire, "sqft_living", "Surface habitable", erreurs, BORNES["sqft_living"]),
        "sqft_lot": lire_entier(formulaire, "sqft_lot", "Surface du terrain", erreurs, BORNES["sqft_lot"]),
        "sqft_basement": lire_entier(formulaire, "sqft_basement", "Surface du sous-sol", erreurs, BORNES["sqft_basement"]),
        "grade": lire_entier(formulaire, "grade", "Qualité de construction", erreurs, choix=LIBELLES_GRADE),
        "condition": lire_entier(formulaire, "condition", "État général", erreurs, choix=LIBELLES_ETAT),
        "view": lire_entier(formulaire, "view", "Qualité de la vue", erreurs, choix=LIBELLES_VUE),
        "zipcode": lire_entier(formulaire, "zipcode", "Code postal", erreurs, choix=ZIPCODES),
        # Cases à cocher : présentes dans le formulaire = cochées
        "waterfront": int("waterfront" in formulaire),
        "renovated": int("renovated" in formulaire),
    }
    annee = lire_entier(formulaire, "annee_construction", "Année de construction", erreurs, BORNES["annee_construction"])
    maison["house_age"] = None if annee is None else ANNEE_REFERENCE - annee

    # Cohérence : le sous-sol fait partie de la surface habitable
    if maison["sqft_living"] is not None and maison["sqft_basement"] is not None \
            and maison["sqft_basement"] >= maison["sqft_living"]:
        erreurs["sqft_basement"] = "Surface du sous-sol : elle doit être inférieure à la surface habitable totale."

    # Emplacement : point cliqué sur la carte, sinon médiane du code postal (calculée sur le train)
    lat = long = None
    if formulaire.get("position_choisie") == "1":
        try:
            lat = lire_coordonnee(formulaire, "lat", BORNES_LAT)
            long = lire_coordonnee(formulaire, "long", BORNES_LONG)
        except ValueError:
            erreurs["carte"] = "Emplacement : le point choisi est en dehors du comté de King. Cliquez sur la carte dans la zone couverte."
    if lat is None or long is None:
        if maison["zipcode"] is not None:
            lat, long = ZIPCODE_LATLONG[maison["zipcode"]]["lat"], ZIPCODE_LATLONG[maison["zipcode"]]["long"]
    maison["lat"], maison["long"] = lat, long

    return maison, erreurs


def estimer_prix(maison):
    """Prédit le prix en dollars : colonnes dans l'ordre du modèle, puis retour de log1p vers les dollars."""
    X = pd.DataFrame([maison])[FEATURES]
    return float(np.expm1(MODELE.predict(X)[0]))


def afficher_page(valeurs, erreurs=None, resultat=None, statut=200):
    """Rend la page avec le formulaire (valeurs conservées), les erreurs éventuelles et le résultat."""
    zipcode = valeurs.get("zipcode", "")
    centre = ZIPCODE_LATLONG.get(int(zipcode) if zipcode.isdigit() else None, ZIPCODE_LATLONG[int(VALEURS_PAR_DEFAUT["zipcode"])])
    contexte = {
        "valeurs": valeurs,
        "erreurs": erreurs or {},
        "resultat": resultat,
        "zipcodes": ZIPCODES,
        "libelles_grade": LIBELLES_GRADE,
        "libelles_vue": LIBELLES_VUE,
        "libelles_etat": LIBELLES_ETAT,
        "bornes": BORNES,
        "annee_reference": ANNEE_REFERENCE,
        "erreur_typique": ERREUR_TYPIQUE,
        "zip_latlong": {str(z): [c["lat"], c["long"]] for z, c in ZIPCODE_LATLONG.items()},
        # Position affichée : point choisi par l'utilisateur, sinon centre du code postal
        "lat": valeurs.get("lat") if valeurs.get("position_choisie") == "1" else centre["lat"],
        "long": valeurs.get("long") if valeurs.get("position_choisie") == "1" else centre["long"],
        "position_choisie": valeurs.get("position_choisie") == "1",
    }
    return render_template("index.html", **contexte), statut


@app.get("/")
def accueil():
    return afficher_page(dict(VALEURS_PAR_DEFAUT))


@app.post("/predict")
def predire():
    valeurs = request.form.to_dict()
    maison, erreurs = valider_formulaire(request.form)
    if erreurs:
        return afficher_page(valeurs, erreurs=erreurs, statut=400)

    prix = estimer_prix(maison)
    resultat = {
        "prix": formater_dollars(prix),
        "prix_brut": prix,
        "bas": formater_dollars(prix * (1 - ERREUR_TYPIQUE)),
        "haut": formater_dollars(prix * (1 + ERREUR_TYPIQUE)),
        "marge": f"{ERREUR_TYPIQUE:.0%}",
        "house_age": maison["house_age"],
        "position_carte": request.form.get("position_choisie") == "1",
    }
    return afficher_page(valeurs, resultat=resultat)


if __name__ == "__main__":
    # Debug désactivé par défaut ; FLASK_DEBUG=1 pour l'activer en développement
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
