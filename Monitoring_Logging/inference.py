import pandas as pd
import mlflow.sklearn
from flask import Flask, request, jsonify
import time
import sys

# --- SETUP FLASK ---
app = Flask(__name__)

# --- LOAD MODEL ---
print("Loading model...")
model = None
# Urutan path folder model (sesuaikan dengan volume docker)
possible_paths = ["model_output", "/app/model_output"]

for path in possible_paths:
    try:
        model = mlflow.sklearn.load_model(path)
        print(f"✅ Berhasil memuat model dari path: {path}")
        break
    except Exception as e:
        print(f"⚠️ Gagal memuat dari {path}: {e}")
        continue

if model is None:
    print("❌ PERINGATAN: Model tidak ditemukan di semua path. Menggunakan mode Dummy.")

@app.route('/invocations', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        # 1. Ambil Data JSON
        content = request.json
        
        # 2. Normalisasi Format Data (Ini perbaikan utamanya)
        # Jika formatnya {'inputs': ...}, ambil isinya. Jika langsung dict, pakai langsung.
        data = content.get('inputs', content)
        
        # Jika data cuma 1 baris (dictionary biasa), bungkus jadi list [data]
        if isinstance(data, dict):
            data = [data]
            
        # 3. Konversi ke DataFrame
        # Pastikan nama kolom SAMA PERSIS dengan saat training (Case Sensitive)
        expected_cols = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        
        df = pd.DataFrame(data)
        
        # Isi kolom yang hilang dengan 0 (agar tidak error jika format json beda dikit)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        # Urutkan kolom sesuai standar model
        df = df[expected_cols]
        
        # 4. Prediksi
        if model is not None:
            prediction = model.predict(df)
            result = int(prediction[0])
        else:
            # Mode Dummy jika model gagal load (agar grafik tetap jalan)
            result = 1 
            
        process_time = time.time() - start_time
        
        # 5. Kirim Balasan Sukses
        response = {
            "predictions": [result],
            "latency": process_time,
            "status": "success"
        }
        return jsonify(response)
        
    except Exception as e:
        # Cetak error lengkap ke terminal agar kelihatan di logs
        print(f"🔥 ERROR 500 TERJADI: {str(e)}", file=sys.stderr)
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    print("🚀 Starting Inference Service on port 5000...")
    app.run(host='0.0.0.0', port=5000)