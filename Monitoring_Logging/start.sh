#!/bin/bash

# 1. Jalankan Inference (Flask) di background
# Tanda '&' berarti jalan di belakang layar agar tidak memblokir proses selanjutnya
echo "--- Menyalakan Inference Service (Port 5000) ---"
python inference.py &

# 2. Tunggu 5 detik
# Memberi waktu agar Flask loading model dulu sampai selesai
sleep 5

# 3. Jalankan Traffic Generator & Exporter
# Ini jalan di foreground (depan layar) agar container tidak mati
echo "--- Menyalakan Exporter & Traffic Generator (Port 8000) ---"
python prometheus_exporter.py