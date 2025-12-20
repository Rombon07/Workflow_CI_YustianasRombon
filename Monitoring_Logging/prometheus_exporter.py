import time
import random
import requests
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# --- 1. SETUP METRIKS PROMETHEUS (10 METRIKS) ---
REQUEST_COUNT = Counter('request_count_total', 'Total request')
FAILED_REQUESTS = Counter('failed_requests_total', 'Total request gagal')
LATENCY = Histogram('prediction_latency_seconds', 'Latency model')
LAST_PREDICTION = Gauge('last_prediction_output', 'Hasil prediksi (0/1)')

# Metrik Input (Inversi Fitur)
INPUT_PH = Gauge('input_feature_ph', 'Nilai pH')
INPUT_HARDNESS = Gauge('input_feature_hardness', 'Nilai Hardness')
INPUT_SOLIDS = Gauge('input_feature_solids', 'Nilai Solids')
INPUT_CHLORAMINES = Gauge('input_feature_chloramines', 'Nilai Chloramines')
INPUT_SULFATE = Gauge('input_feature_sulfate', 'Nilai Sulfate')
INPUT_CONDUCTIVITY = Gauge('input_feature_conductivity', 'Nilai Conductivity')

# URL Inference Service (Localhost karena satu container/network)
INFERENCE_URL = "http://localhost:5000/invocations"

def generate_dummy_data():
    # Generate data random yang masuk akal
    return {
        "ph": random.uniform(0, 14),
        "Hardness": random.uniform(100, 300),
        "Solids": random.uniform(10000, 30000),
        "Chloramines": random.uniform(4, 10),
        "Sulfate": random.uniform(250, 400),
        "Conductivity": random.uniform(300, 600),
        "Organic_carbon": random.uniform(10, 20),
        "Trihalomethanes": random.uniform(50, 90),
        "Turbidity": random.uniform(2, 5)
    }

def monitor_loop():
    print("Starting Prometheus Exporter & Traffic Generator...")
    start_http_server(8000) # Expose Metrics di Port 8000
    
    while True:
        try:
            # 1. Siapkan Data
            payload = generate_dummy_data()
            
            # 2. Update Metrik Input (Feature Inversion langsung di sini)
            INPUT_PH.set(payload['ph'])
            INPUT_HARDNESS.set(payload['Hardness'])
            INPUT_SOLIDS.set(payload['Solids'])
            INPUT_CHLORAMINES.set(payload['Chloramines'])
            INPUT_SULFATE.set(payload['Sulfate'])
            INPUT_CONDUCTIVITY.set(payload['Conductivity'])
            
            # 3. Kirim Request ke Inference.py
            response = requests.post(INFERENCE_URL, json=payload)
            
            # 4. Update Metrik Response
            if response.status_code == 200:
                REQUEST_COUNT.inc()
                data = response.json()
                LAST_PREDICTION.set(data['predictions'][0])
                LATENCY.observe(data.get('latency', 0.1))
                print(f"Sent: pH={payload['ph']:.2f} | Pred: {data['predictions'][0]}")
            else:
                FAILED_REQUESTS.inc()
                print("Request Failed!")
                
        except Exception as e:
            FAILED_REQUESTS.inc()
            print(f"Connection Error (Is inference.py running?): {e}")
            
        time.sleep(2) # Kirim data setiap 2 detik

if __name__ == '__main__':
    monitor_loop()