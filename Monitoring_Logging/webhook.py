from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Baca panjang data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # 2. Tampilkan pesan di terminal
        print('\n\n--- [PESAN DARI GRAFANA MASUK] ---')
        print(post_data.decode('utf-8'))
        print('----------------------------------\n')

        # 3. Kirim Respon Sukses (200 OK)
        self.send_response(200)
        self.end_headers()  # <--- INI YANG TADI KURANG
        self.wfile.write(b"OK")

if __name__ == '__main__':
    print('Server Monitoring siap di Port 7000...')
    # Bind ke 0.0.0.0 agar bisa diakses dari luar container docker
    server = HTTPServer(('0.0.0.0', 7000), SimpleHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer dimatikan.')
        server.server_close()