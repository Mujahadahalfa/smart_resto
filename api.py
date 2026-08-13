#!/usr/bin/env python3

# ============================================
# api.py - REST API untuk Analisis Sentimen
# Smart Resto
#
# Jalankan:
#   python api.py
#
# Port default:
#   5000
# ============================================

import json
import sys
import os

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# Memastikan folder project dapat ditemukan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediksi import prediksi, load_model
from training import main as run_training


# ============================================
# KONFIGURASI SERVER
# ============================================

# 0.0.0.0 diperlukan agar dapat diakses oleh
# server Render dari luar container.
HOST = '0.0.0.0'

# Render menyediakan PORT melalui environment variable.
# Jika dijalankan secara lokal, otomatis menggunakan port 5000.
PORT = int(os.environ.get('PORT', 5000))


# ============================================
# API HANDLER
# ============================================

class SentimenAPIHandler(BaseHTTPRequestHandler):

    # ----------------------------------------
    # Custom log
    # ----------------------------------------

    def log_message(self, format, *args):
        print(
            f"[API] {self.address_string()} - "
            f"{format % args}"
        )

    # ----------------------------------------
    # Response JSON
    # ----------------------------------------

    def send_json(self, data: dict, status: int = 200):

        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode('utf-8')

        self.send_response(status)

        self.send_header(
            'Content-Type',
            'application/json; charset=utf-8'
        )

        self.send_header(
            'Content-Length',
            str(len(body))
        )

        # CORS agar PHP Smart Resto
        # dapat mengakses API
        self.send_header(
            'Access-Control-Allow-Origin',
            '*'
        )

        self.end_headers()

        self.wfile.write(body)

    # ----------------------------------------
    # OPTIONS - CORS
    # ----------------------------------------

    def do_OPTIONS(self):

        self.send_response(200)

        self.send_header(
            'Access-Control-Allow-Origin',
            '*'
        )

        self.send_header(
            'Access-Control-Allow-Methods',
            'GET, POST, OPTIONS'
        )

        self.send_header(
            'Access-Control-Allow-Headers',
            'Content-Type'
        )

        self.end_headers()

    # ========================================
    # GET REQUEST
    # ========================================

    def do_GET(self):

        parsed = urlparse(self.path)

        path = parsed.path

        params = parse_qs(parsed.query)

        # ------------------------------------
        # GET /status
        # ------------------------------------

        if path == '/status':

            try:

                model = load_model()

                self.send_json({
                    'status': 'ok',
                    'model': (
                        'loaded'
                        if model
                        else 'not_trained'
                    ),
                    'akurasi': (
                        model.get('accuracy', 0)
                        if model
                        else 0
                    ),
                    'n_training': (
                        model.get('n_training', 0)
                        if model
                        else 0
                    )
                })

            except Exception as e:

                self.send_json({
                    'status': 'error',
                    'error': str(e)
                }, 500)

        # ------------------------------------
        # GET /prediksi?teks=...
        # ------------------------------------

        elif path == '/prediksi':

            teks = params.get(
                'teks',
                ['']
            )[0]

            if not teks:

                self.send_json({
                    'error': 'Parameter teks diperlukan'
                }, 400)

                return

            try:

                hasil = prediksi(teks)

                self.send_json(hasil)

            except Exception as e:

                self.send_json({
                    'error': str(e)
                }, 500)

        # ------------------------------------
        # GET /info
        # ------------------------------------

        elif path == '/info':

            try:

                model = load_model()

                self.send_json({

                    'app':
                        'Smart Resto Sentiment API',

                    'version':
                        '1.0.0',

                    'algoritma':
                        'Random Forest',

                    'fitur':
                        'TF-IDF',

                    'bahasa':
                        'Indonesia',

                    'endpoint': {

                        'GET /status':
                            'Status model',

                        'GET /prediksi?teks=X':
                            'Prediksi 1 teks',

                        'POST /prediksi':
                            'Prediksi batch (JSON)',

                        'POST /training':
                            'Retrain model'
                    },

                    'model': {

                        'status': (
                            'loaded'
                            if model
                            else 'not_trained'
                        ),

                        'akurasi': (
                            model.get('accuracy', 0)
                            if model
                            else 0
                        ),

                        'evaluasi': (
                            model.get('evaluasi', {})
                            if model
                            else {}
                        )
                    }

                })

            except Exception as e:

                self.send_json({
                    'error': str(e)
                }, 500)

        # ------------------------------------
        # Endpoint tidak ditemukan
        # ------------------------------------

        else:

            self.send_json({
                'error':
                    'Endpoint tidak ditemukan'
            }, 404)

    # ========================================
    # POST REQUEST
    # ========================================

    def do_POST(self):

        parsed = urlparse(self.path)

        path = parsed.path

        # ------------------------------------
        # Baca body
        # ------------------------------------

        length = int(
            self.headers.get(
                'Content-Length',
                0
            )
        )

        body = (
            self.rfile
            .read(length)
            .decode('utf-8')
            if length > 0
            else '{}'
        )

        # ------------------------------------
        # Parse JSON
        # ------------------------------------

        try:

            data = json.loads(body)

        except json.JSONDecodeError:

            self.send_json({
                'error':
                    'Body JSON tidak valid'
            }, 400)

            return

        # ====================================
        # POST /prediksi
        # ====================================

        if path == '/prediksi':

            teks_list = data.get(
                'teks',
                []
            )

            # Jika hanya satu teks
            if isinstance(
                teks_list,
                str
            ):

                teks_list = [
                    teks_list
                ]

            if not teks_list:

                self.send_json({
                    'error':
                        'Field teks diperlukan'
                }, 400)

                return

            try:

                hasil = []

                for teks in teks_list:

                    hasil.append({

                        'teks': teks,

                        'hasil':
                            prediksi(teks)

                    })

                self.send_json({

                    'total':
                        len(hasil),

                    'results':
                        hasil

                })

            except Exception as e:

                self.send_json({
                    'error': str(e)
                }, 500)

        # ====================================
        # POST /training
        # ====================================

        elif path == '/training':

            try:

                model_data = run_training()

                self.send_json({

                    'status':
                        'success',

                    'akurasi':
                        model_data.get(
                            'accuracy',
                            0
                        ),

                    'n_training':
                        model_data.get(
                            'n_training',
                            0
                        ),

                    'message':
                        'Model berhasil dilatih ulang'

                })

            except Exception as e:

                self.send_json({

                    'status':
                        'error',

                    'error':
                        str(e)

                }, 500)

        # ====================================
        # Endpoint tidak ditemukan
        # ====================================

        else:

            self.send_json({

                'error':
                    'Endpoint tidak ditemukan'

            }, 404)


# ============================================
# MENJALANKAN SERVER
# ============================================

def run_server():

    server = HTTPServer(
        (HOST, PORT),
        SentimenAPIHandler
    )

    print(
        "\n" +
        "=" * 50
    )

    print(
        "  Smart Resto Sentiment Analysis API"
    )

    print(
        f"  Berjalan di: http://{HOST}:{PORT}"
    )

    print(
        "  Endpoint tersedia:"
    )

    print(
        "    GET  /status"
    )

    print(
        "    GET  /info"
    )

    print(
        "    GET  /prediksi?teks=..."
    )

    print(
        "    POST /prediksi"
    )

    print(
        "    POST /training"
    )

    print(
        "  Server siap menerima request."
    )

    print(
        "=" * 50 +
        "\n"
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\n[API] Server dihentikan."
        )

    finally:

        server.server_close()


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':

    run_server()
