import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import shutil
import logging

# --- 1. SETUP PEREDAM SUARA (Supaya Terminal Bersih) ---
# Matikan log 'Info' MLflow yang suka bikin merah di PowerShell
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

# Setup Path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"[INFO] Direktori Kerja: {os.getcwd()}")

# Bersih-bersih folder temp
if os.path.exists("temp_model_storage"):
    shutil.rmtree("temp_model_storage")

# Load Data
try:
    df = pd.read_csv('water_potability_clean.csv')
except FileNotFoundError:
    print("[ERROR] File CSV tidak ditemukan.")
    exit()

X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. SETUP MLFLOW ---
# Kita set experiment baru lagi biar fresh
mlflow.set_experiment("ML_RF_Modelling_Yustianas_Rombon")

print("[INFO] Memulai Tuning & Upload Paksa...")

# --- 3. PROSES TUNING ---
with mlflow.start_run(run_name="Proses_Tuning_Final") as parent_run:
    
    # A. Tuning Singkat
    rf = RandomForestClassifier(random_state=42)
    # 1 Iterasi saja biar cepat selesai
    param_dist = {'n_estimators': [50], 'max_depth': [10]} 
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=1, cv=2, random_state=42)
    random_search.fit(X_train, y_train)
    
    # B. Generate Model Files di Folder Lokal
    print("[INFO] Sedang men-generate file model di lokal...")
    best_model = random_search.best_estimator_
    
    # Simpan ke folder sementara
    mlflow.sklearn.save_model(best_model, "temp_model_storage")
    
    # C. UPLOAD FOLDER MODEL KE ARTIFACT (CARA PAKSA)
    print("[INFO] Sedang meng-upload folder model ke Artifact Browser...")
    mlflow.log_artifacts("temp_model_storage", artifact_path="model")
    
    # D. Log Gambar Confusion Matrix
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    
    mlflow.log_artifact("training_confusion_matrix.png")
    mlflow.log_metric("accuracy", acc)
    
    # E. Hapus folder temp
    if os.path.exists("temp_model_storage"):
        shutil.rmtree("temp_model_storage")
        
    print(f"[SUKSES] Folder 'model' berhasil masuk ke Artifacts! Silakan cek UI.")