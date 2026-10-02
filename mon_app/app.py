"""
Serveur Flask — Détection de maladies de plantes (YOLOv8, modèle Roboflow)
avec recommandations de traitement intégrées.

Usage :
    pip install -r requirements_server.txt
    python app.py --model best_cropdisease.pt

Puis ouvre http://localhost:5000 (ou http://<IP-machine>:5000 depuis un téléphone
sur le même réseau Wi-Fi).
"""

import argparse
import io
import json
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

model = None
advice_db = {}


def load_advice(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"⚠️  Fichier de conseils introuvable : {path} (les recommandations seront absentes des réponses)")
    return {}


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'advice_entries': len(advice_db),
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé côté serveur.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': "Aucune image fournie. Utilise le champ 'image' (multipart/form-data)."}), 400

    file = request.files['image']

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

        # Ajoute les conseils si disponibles pour cette classe
        if cls_name in advice_db:
            entry['advice'] = advice_db[cls_name]

        detections.append(entry)

    detections.sort(key=lambda d: d['confidence'], reverse=True)

    return jsonify({
        'detections': detections,
        'alert': len(detections) > 0,
        'count': len(detections),
    })


def main():
    global model, advice_db
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_cropdisease.pt')
    parser.add_argument('--advice_file', type=str, default='disease_advice.json')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Modèle introuvable : {args.model}")
        print("   Vérifie que best_cropdisease.pt est bien dans ce dossier, ou précise --model <chemin>.")
        return

    print(f"Chargement du modèle depuis : {args.model}")
    model = YOLO(args.model)
    print(f"Modèle chargé. Classes : {len(model.names)}")

    advice_db = load_advice(args.advice_file)
    print(f"Base de conseils chargée : {len(advice_db)} entrées")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        ssl_context=('cert.pem', 'key.pem')
    )

if __name__ == '__main__':
    main()