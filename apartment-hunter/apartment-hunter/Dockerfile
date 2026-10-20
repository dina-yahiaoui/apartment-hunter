FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dépendances d'abord : cette couche est mise en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Fichiers de l'application (code, templates, modèle entraîné)
COPY app.py .
COPY templates/ templates/
COPY model/ model/

# Utilisateur non-root
RUN useradd --create-home appuser
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# 3) Commande de démarrage : gunicorn (serveur de production) lance l'application Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
