#!/usr/bin/env python3
# ============================================
# api.py - REST API untuk Analisis Sentimen
# Smart Resto
# Jalankan: python3 api.py
# Port default: 5000
# ============================================

import json
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(__file__))
from prediksi import prediksi, load_model
from training import main as run_training

HOST = 'localhost'
PORT = 5000


class SentimenAPIHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Custom log format
        print(f"[API] {self.address_string()} - {format % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        # ── GET /status ──
        if path == '/status':
            model = load_model()
            self.send_json({
                'status':  'ok',
                'model':   'loaded' if model else 'not_trained',
                'akurasi': model.get('accuracy', 0) if model else 0,
                'n_training': model.get('n_training', 0) if model else 0,
            })

        # ── GET /prediksi?teks=... ──
        elif path == '/prediksi':
            teks = params.get('teks', [''])[0]
            if not teks:
                self.send_json({'error': 'Parameter teks diperlukan'}, 400)
                return
            hasil = prediksi(teks)
            self.send_json(hasil)

        # ── GET /info ──
        elif path == '/info':
            model = load_model()
            self.send_json({
                'app':        'Smart Resto Sentiment API',
                'version':    '1.0.0',
                'algoritma':  'Random Forest',
                'fitur':      'TF-IDF',
                'bahasa':     'Indonesia',
                'endpoint': {
                    'GET /status':          'Status model',
                    'GET /prediksi?teks=X': 'Prediksi 1 teks',
                    'POST /prediksi':       'Prediksi batch (JSON)',
                    'POST /training':       'Retrain model',
                },
                'model': {
                    'status':   'loaded' if model else 'not_trained',
                    'akurasi':  model.get('accuracy', 0) if model else 0,
                    'evaluasi': model.get('evaluasi', {}) if model else {},
                }
            })

        else:
            self.send_json({'error': 'Endpoint tidak ditemukan'}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        # Baca body
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({'error': 'Body JSON tidak valid'}, 400)
            return

        # ── POST /prediksi ──
        if path == '/prediksi':
            # Batch prediction
            teks_list = data.get('teks', [])
            if isinstance(teks_list, str):
                teks_list = [teks_list]

            if not teks_list:
                self.send_json({'error': 'Field teks diperlukan'}, 400)
                return

            hasil = []
            for teks in teks_list:
                hasil.append({'teks': teks, 'hasil': prediksi(teks)})

            self.send_json({
                'total':   len(hasil),
                'results': hasil
            })

        # ── POST /training ──
        elif path == '/training':
            try:
                model_data = run_training()
                self.send_json({
                    'status':   'success',
                    'akurasi':  model_data.get('accuracy', 0),
                    'n_training': model_data.get('n_training', 0),
                    'message':  'Model berhasil dilatih ulang'
                })
            except Exception as e:
                self.send_json({'error': str(e)}, 500)

        else:
            self.send_json({'error': 'Endpoint tidak ditemukan'}, 404)


def run_server():
    server = HTTPServer((HOST, PORT), SentimenAPIHandler)
    print(f"\n{'='*50}")
    print(f"  Smart Resto Sentiment Analysis API")
    print(f"  Berjalan di: http://{HOST}:{PORT}")
    print(f"  Endpoint tersedia:")
    print(f"    GET  /status")
    print(f"    GET  /info")
    print(f"    GET  /prediksi?teks=...")
    print(f"    POST /prediksi")
    print(f"    POST /training")
    print(f"  Tekan Ctrl+C untuk berhenti")
    print(f"{'='*50}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[API] Server dihentikan.")
        server.server_close()


if __name__ == '__main__':
    run_server()
