"""
Serveur Flask — Architecture caméra/tableau de bord séparés.

Le téléphone envoie des images (route /predict), le PC consulte le
dernier résultat (route /latest) et l'historique (route /history).
Chaque capture est sauvegardée en base SQLite + image sur disque.

Usage :
    pip install -r requirements_server.txt
    python app.py --model best_cropdisease.pt

Sur le téléphone : http://<IP>:5000/camera
Sur le PC        : http://<IP>:5000/   (ou http://localhost:5000/)
"""

import argparse
import io
import json
import os
import sqlite3
import time
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

model = None
advice_db = {}

DB_PATH = 'captures.db'
CAPTURES_DIR = os.path.join('static', 'captures')


# ----------------------- Base de données -----------------------

def init_db():
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detections_json TEXT NOT NULL,
            alert INTEGER NOT NULL,
            latitude REAL,
            longitude REAL
        )
    ''')
    conn.commit()
    conn.close()


def save_capture(image, detections, latitude=None, longitude=None):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{timestamp}_{int(time.time() * 1000) % 1000}.jpg"
    filepath = os.path.join(CAPTURES_DIR, filename)
    image.save(filepath, 'JPEG', quality=85)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO captures (timestamp, image_path, detections_json, alert, latitude, longitude) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (
            datetime.now().isoformat(),
            f"captures/{filename}",
            json.dumps(detections),
            1 if len(detections) > 0 else 0,
            latitude,
            longitude,
        )
    )
    conn.commit()
    conn.close()


def get_latest():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM captures ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return dict(row) if row else None


def get_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM captures ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ----------------------- Pages web -----------------------

@app.route('/', methods=['GET'])
def dashboard():
    return send_from_directory(app.static_folder, 'dashboard.html')


@app.route('/camera', methods=['GET'])
def camera_page():
    return send_from_directory(app.static_folder, 'camera.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'advice_entries': len(advice_db),
    })


# ----------------------- API -----------------------

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé côté serveur.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': "Aucune image fournie (champ 'image' attendu)."}), 400

    file = request.files['image']
    latitude = request.form.get('latitude', type=float)
    longitude = request.form.get('longitude', type=float)

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Image illisible : {e}'}), 400

    img_w, img_h = image.size
    results = model(image, verbose=False, conf=0.25)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        entry = {
            'class': cls_name,
            'confidence': round(conf * 100, 2),
            'box': {
                'x1': x1 / img_w, 'y1': y1 / img_h,
                'x2': x2 / img_w, 'y2': y2 / img_h,
            }
        }
        if cls_name in advice_db:
            entry['advice'] = advice_db[cls_name]
        detections.append(entry)

    detections.sort(key=lambda d: d['confidence'], reverse=True)

    # Sauvegarde systématique (image + résultat), même si aucune détection
    save_capture(image, detections, latitude, longitude)

    return jsonify({
        'detections': detections,
        'alert': len(detections) > 0,
        'count': len(detections),
    })


@app.route('/latest', methods=['GET'])
def latest():
    row = get_latest()
    if row is None:
        return jsonify({'empty': True})

    row['detections'] = json.loads(row['detections_json'])
    del row['detections_json']
    return jsonify(row)


@app.route('/history', methods=['GET'])
def history():
    limit = request.args.get('limit', default=20, type=int)
    rows = get_history(limit)
    for row in rows:
        row['detections'] = json.loads(row['detections_json'])
        del row['detections_json']
    return jsonify(rows)


# ----------------------- Démarrage -----------------------

def load_advice(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"⚠️  Fichier de conseils introuvable : {path}")
    return {}


def main():
    global model, advice_db
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_cropdisease.pt')
    parser.add_argument('--advice_file', type=str, default='disease_advice.json')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--https', action='store_true',
                         help="Active HTTPS (nécessite cert.pem et key.pem dans ce dossier)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Modèle introuvable : {args.model}")
        return

    init_db()
    print("Base de données initialisée :", DB_PATH)

    print(f"Chargement du modèle depuis : {args.model}")
    model = YOLO(args.model)
    print(f"Modèle chargé. Classes : {len(model.names)}")

    advice_db = load_advice(args.advice_file)
    print(f"Base de conseils chargée : {len(advice_db)} entrées")

    ssl_context = None
    if args.https:
        if not (os.path.exists('cert.pem') and os.path.exists('key.pem')):
            print("❌ --https demandé mais cert.pem / key.pem introuvables dans ce dossier.")
            print("   Génère-les avec :")
            print("   openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj \"/CN=<TON_IP>\"")
            return
        ssl_context = ('cert.pem', 'key.pem')
        print("HTTPS activé (certificat auto-signé — un avertissement du navigateur est normal, à accepter manuellement).")

    app.run(host=args.host, port=args.port, debug=args.debug, ssl_context=ssl_context)


if __name__ == '__main__':
    main()