# Używamy oficjalnego obrazu Pythona
FROM python:3.12

# Ustawiamy katalog roboczy
WORKDIR /app

# Kopiujemy pliki do kontenera
COPY . /app

# Instalujemy wymagane pakiety
RUN pip install --no-cache-dir -r requirements.txt

# Otwieramy port, na którym działa FastAPI
EXPOSE 8000

# Uruchamiamy API FastAPI za pomocą Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]



