import time
import pandas as pd
import mlflow.sklearn
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# --- 1. SETUP PROMETHEUS METRICS (SYARAT ADVANCE: MINIMAL 10 METRIKS) ---
# Metrik Operasional
REQUEST_COUNT = Counter('request_count_total', 'Total jumlah request yang masuk')
FAILED_REQUESTS = Counter('failed_requests_total', 'Total request yang gagal/error')
LATENCY = Histogram('prediction_latency_seconds', 'Waktu yang dibutuhkan untuk prediksi')
LAST_PREDICTION = Gauge('last_prediction_output', 'Hasil prediksi terakhir (0=Tidak Layak, 1=Layak)')

# Metrik Data Drift (Memantau Distribusi Input Fitur)
INPUT_PH = Gauge('input_feature_ph', 'Nilai input pH air')
INPUT_HARDNESS = Gauge('input_feature_hardness', 'Nilai input Hardness air')
INPUT_SOLIDS = Gauge('input_feature_solids', 'Nilai input Solids air')
INPUT_CHLORAMINES = Gauge('input_feature_chloramines', 'Nilai input Chloramines air')
INPUT_SULFATE = Gauge('input_feature_sulfate', 'Nilai input Sulfate air')
INPUT_CONDUCTIVITY = Gauge('input_feature_conductivity', 'Nilai input Conductivity air')

# --- 2. LOAD MODEL ---
# Pastikan folder 'model_output' hasil training sebelumnya ada di sebelah file ini
# Atau ganti path sesuai lokasi model Anda
print("Loading model...")
try:
    model = mlflow.sklearn.load_model("../model_output") # Path ke model lokal
except:
    print("Model lokal tidak ditemukan. Pastikan path benar.")
    model = None

# --- 3. SETUP FLASK APP ---
app = Flask(__name__)

@app.route('/invocations', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc() # Tambah counter request
    
    try:
        content = request.json
        # Format input bisa berbeda tergantung client, kita handle list of lists
        data = content['inputs'] if 'inputs' in content else content
        
        df = pd.DataFrame(data, columns=[
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ])
        
        # Update Metrik Fitur (Ambil data baris pertama)
        INPUT_PH.set(df['ph'].iloc[0])
        INPUT_HARDNESS.set(df['Hardness'].iloc[0])
        INPUT_SOLIDS.set(df['Solids'].iloc[0])
        INPUT_CHLORAMINES.set(df['Chloramines'].iloc[0])
        INPUT_SULFATE.set(df['Sulfate'].iloc[0])
        INPUT_CONDUCTIVITY.set(df['Conductivity'].iloc[0])

        # Prediksi
        prediction = model.predict(df)
        result = int(prediction[0])
        
        # Update Metrik Hasil
        LAST_PREDICTION.set(result)
        
        # Hitung Latency
        process_time = time.time() - start_time
        LATENCY.observe(process_time)
        
        return jsonify({"predictions": [result]})
        
    except Exception as e:
        FAILED_REQUESTS.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan Server Metrik Prometheus di Port 8000
    print("Starting Prometheus metrics server on port 8000...")
    start_http_server(8000)
    
    # Jalankan Aplikasi Flask di Port 5000
    print("Starting Flask app on port 5000...")
    app.run(host='0.0.0.0', port=5000)