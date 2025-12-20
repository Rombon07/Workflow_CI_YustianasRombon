FROM python:3.9-slim

WORKDIR /app

# 1. Install Dependencies
# Pastikan requirements.txt sudah berisi 'requests'
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy SEMUA file codingan ke dalam container
# (Termasuk inference.py, prometheus_exporter.py, dan start.sh)
COPY . .

# 3. Berikan Izin Eksekusi ke script start.sh
# Agar script bisa dijalankan oleh Linux
RUN chmod +x start.sh

# 4. Buka Port
# 5000 untuk API Model, 8000 untuk Metrics Prometheus
EXPOSE 5000 8000

# 5. Perintah Menjalankan
# Saat container nyala, jalankan script start.sh
CMD ["./start.sh"]