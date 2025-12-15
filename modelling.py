import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil # Tambahan untuk bersih-bersih folder

# --- 1. KONFIGURASI DAGSHUB ---
DAGSHUB_USERNAME = 'yustianasrombon7'
DAGSHUB_REPO_NAME = 'Eksperimen_SML_YustianasRombon'

if not os.getenv("MLFLOW_TRACKING_URI"):
    print("Running Lokal: Menginisialisasi DagsHub secara interaktif...")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
else:
    print("Running di CI/CD: Menggunakan Environment Variables.")

mlflow.set_experiment("Water Potability CI Pipeline")

def main():
    # --- 2. Load Data ---
    print("Memuat data...")
    try:
        df = pd.read_csv('water_potability_clean.csv')
    except FileNotFoundError:
        print("File tidak ditemukan, mencoba path absolut...")
        df = pd.read_csv(os.path.join(os.getcwd(), 'water_potability_clean.csv'))
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Training ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=20, min_samples_split=5)
    
    # Kita skip GridSearch biar cepat debugging (Langsung fit)
    # Kalau mau GridSearch lagi, silakan uncomment kode lama.
    # Untuk debugging CI/CD, keep it simple dulu.
    
    with mlflow.start_run(run_name="Fixed_CI_Run") as run:
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model") # Tetap log ke DagsHub (Dashboard)
        
        # --- PERBAIKAN UTAMA: SIMPAN LOKAL UNTUK DOCKER ---
        # Hapus folder lama jika ada
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
            
        print("Menyimpan model ke folder lokal 'model_output' untuk Docker...")
        mlflow.sklearn.save_model(rf, "model_output")
        
        print("Selesai.")

if __name__ == "__main__":
    main()