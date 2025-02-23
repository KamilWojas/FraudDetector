Fraud Detection AI 

 Opis projektu -

System do wykrywania oszustw oparty na Machine Learning i AI. Projekt analizuje transakcje w celu identyfikacji potencjalnych oszustw i wyłudzeń.

###  Technologie i narzędzia
- **Język:** Python 
- **Analiza danych:** Pandas, NumPy  
- **Wizualizacja:** Matplotlib, Seaborn  
- **Uczenie maszynowe:** Scikit-learn, XGBoost, TensorFlow/PyTorch  
- **Big Data:** Apache Spark, Dask  
- **API:** Flask/FastAPI  
- **Deployment:** Docker, Google Cloud  



 Struktura projektu
FraudDetector/
├── data/               #  Zbiory danych (surowe, przetworzone, wyniki)
│   ├── raw/            # Surowe dane wejściowe
│   ├── processed/      # Dane po czyszczeniu i przetwarzaniu
│   ├── results/        # Wyniki predykcji modelu
│   ├── transactions.csv  # Przykładowy plik z transakcjami
│
├── notebooks/          #  Jupyter Notebooki do eksploracji danych i testów modeli
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│
├── src/               #  Kod źródłowy
│   ├── preprocessing.py   # Skrypty do czyszczenia i przygotowania danych
│   ├── feature_engineering.py  # Tworzenie cech dla modelu
│   ├── model.py          # Trening i testowanie modelu
│   ├── api.py            # API do wykrywania oszustw (Flask / FastAPI)
│   ├── utils.py          # Funkcje pomocnicze
│
├── tests/              #  Testy jednostkowe kodu
│   ├── test_preprocessing.py
│   ├── test_model.py
│
├── deployment/         #  Skrypty do wdrożenia modelu
│   ├── Dockerfile      # Plik do uruchamiania projektu w Dockerze
│   ├── requirements.txt  # Lista wymaganych bibliotek
│   ├── config.yaml      # Konfiguracja modelu
│
├── README.md           #  Dokumentacja projektu
├── .gitignore          #  Plik ignorowania w Git (np. pliki danych, logi)
├── requirements.txt    #  Lista zależności do instalacji
├── setup.py            #  Skrypt do pakowania projektu



