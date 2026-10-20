# Image de base : Python 3.13 (même version que le .venv d'entraînement), variante « slim » allégée
FROM python:3.13-slim

# Pas de fichiers .pyc et logs affichés immédiatement (utile pour `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dossier de travail dans le conteneur : le projet vivra dans /app
WORKDIR /app

# 1. Copie des seules dépendances d'abord : tant que requirements.txt ne change pas,
#    Docker réutilise cette couche en cache et ne réinstalle pas les paquets à chaque build
COPY app/requirements.txt ./app/requirements.txt

# 2. Installation des dépendances sans garder le cache de pip (image plus légère)
RUN pip install --no-cache-dir -r app/requirements.txt

# 3. Copie du code de l'application et des fichiers du modèle :
#    /app/app/app.py retrouve /app/models/ via pathlib (dossier parent de app/)
COPY app/ ./app/
COPY models/ ./models/

# Exécution avec un utilisateur sans privilèges plutôt que root (bonne pratique de sécurité)
RUN useradd --create-home appuser
USER appuser

# Port sur lequel l'application écoute dans le conteneur
EXPOSE 5000

# Lancement avec gunicorn (serveur de production) : 2 processus, à l'écoute sur toutes les interfaces ;
# --chdir app place gunicorn dans /app/app pour importer l'objet `app` du module app.py
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--chdir", "app", "app:app"]
