# Sections à intégrer au README.md (étapes 8 à 12)

## Sélection des variables (étape 8) — `notebooks/01_selection_features.ipynb`
Quatre méthodes votent (information mutuelle, Lasso règle 1-SE, forward selection, Boruta) ; le VIF contrôle la redondance. Une variable de bruit pur (`random_noise`) sert de témoin. Une variable est retenue avec au moins 2 votes sur 4, puis on retire les variables redondantes (VIF > 10 : `sqft_above`, `yr_built`, `sqft_basement`).
**16 variables finales :** `sqft_living`, `view`, `sqft_lot`, `sqft_living15`, `lat`, `grade`, `house_age`, `waterfront`, `bathrooms`, `floors`, `has_basement`, `yr_renovated`, `condition`, `sqft_lot15`, `long`, `bedrooms`.

## Modélisation (étapes 9-10) — `notebooks/02_modelisation.ipynb`
Cible : log(prix). Découpage 80/20 groupé par `id` (biens revendus). Comparaison par validation croisée groupée à 5 plis, puis `GridSearchCV` (3 plis).

| Modèle | R² CV (défaut) | R² test (log, optimisé) | Erreur médiane test |
|---|---|---|---|
| Régression linéaire | 0,768 | 0,770 | 15,8 % |
| KNN | 0,834 | 0,849 | 9,9 % |
| Random Forest | 0,886 | 0,892 | 8,6 % |
| Gradient Boosting | 0,897 | 0,904 | 8,4 % |
| **HistGradientBoosting (retenu)** | 0,899 | 0,904 | 8,4 % |

58 % des estimations à ±10 % du prix réel, 83 % à ±20 %. Variables les plus influentes : `lat`, `sqft_living`, `grade`, `long`.
Limites : données de 2014-2015 d'un seul comté, précision moindre sur les biens très bas ou très haut de gamme.

## Application Flask (étape 11) — `app.py`
Formulaire (`/`), API JSON (`POST /api/predict`) et `/health`. Le code postal pré-remplit latitude, longitude et voisinage. L'estimation est accompagnée d'une fourchette à 80 % (erreurs observées sur le test : environ -16 % / +21 %).

```bash
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

## Docker (étape 12) — `Dockerfile`
```bash
docker build -t apartment-hunter .
docker run -p 5000:5000 apartment-hunter     # http://localhost:5000
```

## Structure du repository
```
apartment-hunter/
├── notebooks/ (01_selection_features.ipynb, 02_modelisation.ipynb, variables_finales.csv)
├── data/kc_house_data.csv
├── model/ (model.joblib, model_meta.json)
├── templates/index.html
├── app.py · requirements.txt · Dockerfile · .dockerignore
└── dashboard/  (ton export Power BI)
```
