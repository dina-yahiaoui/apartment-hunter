"""Apartment Hunter — application Flask d'estimation de prix immobilier (King County)."""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / "model" / "model.joblib")
META = json.loads((BASE / "model" / "model_meta.json").read_text(encoding="utf-8"))
FEATURES = META["features"]
ZIPCODES = META["zipcodes"]

app = Flask(__name__)

# (nom, type, minimum, maximum) — limites strictes de validation
SPEC = {
    "sqft_living": (int, 200, 15000),
    "sqft_lot": (int, 300, 2_000_000),
    "bedrooms": (int, 0, 12),
    "bathrooms": (float, 0.5, 8),
    "floors": (float, 1, 3.5),
    "grade": (int, 1, 13),
    "condition": (int, 1, 5),
    "view": (int, 0, 4),
    "yr_built": (int, 1800, date.today().year),
    "yr_renovated": (int, 0, date.today().year),
    "lat": (float, 47.0, 48.0),
    "long": (float, -122.6, -121.0),
    "sqft_living15": (int, 200, 15000),
    "sqft_lot15": (int, 300, 2_000_000),
}
LABELS = {
    "sqft_living": "Surface habitable", "sqft_lot": "Surface du terrain", "bedrooms": "Chambres",
    "bathrooms": "Salles de bain", "floors": "Étages", "grade": "Qualité de construction",
    "condition": "État général", "view": "Vue", "yr_built": "Année de construction",
    "yr_renovated": "Année de rénovation", "lat": "Latitude", "long": "Longitude",
    "sqft_living15": "Surface moyenne des voisins", "sqft_lot15": "Terrain moyen des voisins",
}


def parse_input(raw):
    """Valide les champs reçus (formulaire ou JSON) et construit la ligne de variables du modèle.
    Retourne (ligne | None, valeurs_nettoyées, erreurs, avertissements)."""
    errors, warnings, vals = [], [], {}

    # Le code postal pré-remplit la localisation et le voisinage quand ils ne sont pas fournis
    zc = str(raw.get("zipcode", "")).strip()
    zinfo = ZIPCODES.get(zc)
    if zc and not zinfo:
        errors.append("Code postal inconnu dans la base King County.")
    defaults = {}
    if zinfo:
        defaults = {k: zinfo[k] for k in ("lat", "long", "sqft_living15", "sqft_lot15")}

    for name, (typ, lo, hi) in SPEC.items():
        v = raw.get(name)
        if v in (None, ""):
            if name == "yr_renovated":
                v = 0
            elif name in defaults:
                v = defaults[name]
            elif name in ("sqft_living15", "sqft_lot15"):
                v = vals.get("sqft_living" if name == "sqft_living15" else "sqft_lot")
                if v is None:        # le champ de base est déjà signalé en erreur
                    continue
            elif name in ("lat", "long") and zc and not zinfo:
                continue             # code postal invalide déjà signalé
            else:
                errors.append(f"« {LABELS[name]} » est obligatoire.")
                continue
        try:
            v = typ(float(v))
        except (TypeError, ValueError):
            errors.append(f"« {LABELS[name]} » doit être un nombre.")
            continue
        if name == "yr_renovated" and v == 0:
            pass
        elif not lo <= v <= hi:
            errors.append(f"« {LABELS[name]} » doit être compris entre {lo} et {hi}.")
            continue
        vals[name] = v

    vals["waterfront"] = int(str(raw.get("waterfront", "0")).lower() in ("1", "true", "on", "oui"))
    vals["has_basement"] = int(str(raw.get("has_basement", "0")).lower() in ("1", "true", "on", "oui"))
    vals["zipcode"] = zc
    if errors:
        return None, vals, errors, warnings

    if vals["yr_renovated"] and vals["yr_renovated"] < vals["yr_built"]:
        errors.append("L'année de rénovation ne peut pas précéder l'année de construction.")
        return None, vals, errors, warnings

    # Même définition que dans le notebook : âge à la date de vente (année de référence des données)
    vals["house_age"] = max(META["ref_year"] - vals["yr_built"], 0)

    # Avertissement si une valeur sort de la zone d'entraînement
    for f, (lo, hi) in META["bounds"].items():
        if f in vals and not lo <= vals[f] <= hi:
            warnings.append(f"{LABELS.get(f, f)} hors de la plage observée à l'entraînement "
                            f"({lo:g}–{hi:g}) : l'estimation sera moins fiable.")

    row = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
    return row, vals, errors, warnings


def estimate(row):
    """Prédit en log, convertit en dollars et ajoute une fourchette à 80 %."""
    p = float(np.exp(MODEL.predict(row)[0]))
    low, high = p * np.exp(META["log_resid_q10"]), p * np.exp(META["log_resid_q90"])
    r = lambda x: int(round(x, -3))
    return {"estimate": r(p), "low": r(low), "high": r(high)}


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, warnings, form = None, [], [], {}
    if request.method == "POST":
        form = request.form.to_dict()
        row, vals, errors, warnings = parse_input(form)
        if row is not None:
            result = estimate(row)
    return render_template(
        "index.html", result=result, errors=errors, warnings=warnings, form=form,
        zipcodes=ZIPCODES, meta=META,
    )


@app.post("/api/predict")
def api_predict():
    row, _, errors, warnings = parse_input(request.get_json(silent=True) or {})
    if row is None:
        return jsonify({"errors": errors}), 400
    return jsonify({**estimate(row), "currency": "USD", "prices_year": "2014-2015", "warnings": warnings})


@app.get("/health")
def health():
    return jsonify(status="ok", model=META["model_name"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
