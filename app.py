"""
Retinal Disease Detection System — Flask Backend
"""

import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─── APP CONFIG ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER  = Path("uploads")
DB_FILE         = Path("database/predictions.json")
MODEL_PATH     = Path("model/retinal_model.pt")
ALLOWED_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPLOAD_FOLDER.mkdir(exist_ok=True)
Path("database").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# ─── CLASS METADATA (Updated for Cataract & New Order) ───────────────────────
CLASS_INFO = {
    "cataract": {
        "label":      "Cataract",
        "status":     "ABNORMAL",
        "icon":       "☁️",
        "color":      "#747d8c",
        "description": "Clouding of the eye's natural lens, which leads to blurry vision.",
        "precautions": [
            "Consult an ophthalmologist for surgery evaluation",
            "Use brighter lighting at home",
            "Wear anti-glare sunglasses",
            "Regular eye pressure monitoring",
            "Manage blood sugar if diabetic"
        ],
        "foods": [
            "Blueberries - rich in antioxidants.",
            "Spinach - contains Lutein.",
            "Salmon - Omega-3 for lens health.",
            "Almonds - Vitamin E protection."
        ]
    },
    "diabetic_retinopathy": {
        "label":      "Diabetic Retinopathy",
        "status":     "ABNORMAL",
        "icon":       "🩸",
        "color":      "#ff4757",
        "description": "Damage to the blood vessels in the retinal tissue caused by diabetes mellitus.",
        "precautions": [
            "Control blood sugar levels strictly",
            "Schedule regular ophthalmology checkups every 6 months",
            "Maintain healthy blood pressure (< 130/80 mmHg)",
            "Avoid smoking and excessive alcohol",
            "Consult a retinal specialist immediately"
        ],
        "foods": [
            "Leafy green vegetables (Spinach, Kale) - helps control blood sugar.",
            "Citrus fruits and Berries (Oranges, Strawberries) - rich in Vitamin C.",
            "Nuts and Seeds (Walnuts, Chia seeds) - good for retinal health.",
            "Fatty fish (Salmon, Sardines) - rich in Omega-3 fatty acids."
        ]
    },
    "glaucoma": {
        "label":      "Glaucoma",
        "status":     "ABNORMAL",
        "icon":       "👁",
        "color":      "#ffa502",
        "description": "A group of eye conditions that damage the optic nerve, often caused by elevated intraocular pressure.",
        "precautions": [
            "Use prescribed eye drops consistently",
            "Monitor intraocular pressure regularly",
            "Avoid high-impact activities that raise eye pressure",
            "Sleep with head slightly elevated",
            "Follow up with ophthalmologist every 3–6 months"
        ],
        "foods": [
            "Peaches and Blueberries - contains antioxidants that protect optic nerves.",
            "Dark leafy greens (Spinach, Collard greens) - helps lower eye pressure.",
            "Green tea - rich in flavonoids to protect the eye structure.",
            "Carrots and Sweet potatoes - high in Vitamin A for nerve health."
        ]
    },
    "macular_degeneration": {
        "label":      "Macular Degeneration",
        "status":     "ABNORMAL",
        "icon":       "🔵",
        "color":      "#eccc68",
        "description": "Age-related deterioration of the macula, the central part of the retina responsible for sharp vision.",
        "precautions": [
            "Take AREDS2 nutritional supplements if advised",
            "Use UV-protective sunglasses outdoors",
            "Eat a diet rich in leafy greens and fish",
            "Stop smoking immediately",
            "See a retinal specialist for anti-VEGF injections if needed"
        ],
        "foods": [
            "Carrots and Bell peppers - excellent sources of Beta-carotene.",
            "Eggs - contains Lutein and Zeaxanthin which protects the macula.",
            "Oranges, Kiwis, and Grapefruits - packed with Vitamin C for eye protection.",
            "Almonds and Sunflower seeds - high in Vitamin E to slow degeneration."
        ]
    },
    "normal": {
        "label":      "Normal",
        "status":     "HEALTHY",
        "icon":       "✅",
        "color":      "#2ed573",
        "description": "No signs of retinal disease detected. The fundus appears healthy.",
        "precautions": [
            "Continue routine eye exams annually",
            "Maintain a balanced diet rich in vitamins A, C, E",
            "Protect eyes from UV light with sunglasses",
            "Limit screen time and practice the 20-20-20 rule",
            "Stay physically active to support overall eye health"
        ],
        "foods": [] 
    }
}

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
model       = None
class_names = list(CLASS_INFO.keys())

def load_model():
    global model, class_names
    if not MODEL_PATH.exists():
        print(f"⚠  Model not found at {MODEL_PATH} — running in DEMO mode")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint.get('classes', class_names)

    base = models.resnet50(weights=None)
    in_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    base.load_state_dict(checkpoint['model_state'])
    base.eval()
    model = base.to(DEVICE)
    print(f"✓ Model loaded  |  Classes: {class_names}  |  Device: {DEVICE}")

load_model()

# ─── INFERENCE ───────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path: str):
    if model is None:
        import random
        key   = random.choice(class_names)
        probs = {c: round(random.uniform(0.02, 0.15), 4) for c in class_names}
        probs[key] = round(random.uniform(0.60, 0.95), 4)
        total = sum(probs.values())
        probs = {c: round(v / total, 4) for c, v in probs.items()}
        return key, round(probs[key] * 100, 2), probs

    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    key = class_names[idx]
    all_probs = {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}
    return key, round(float(probs[idx]) * 100, 2), all_probs

# ─── JSON DATABASE ────────────────────────────────────────────────────────────
def db_load():
    if DB_FILE.exists():
        with open(DB_FILE) as f:
            return json.load(f)
    return {"predictions": []}

def db_save(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def db_insert(record: dict):
    data = db_load()
    data["predictions"].insert(0, record)
    db_save(data)

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # ─── UPDATE: Get patient details from frontend ───
    patient_name = request.form.get("patient_name", "Guest")
    patient_age = request.form.get("patient_age", "N/A")
    # ────────────────────────────────────────────────

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 415

    uid       = uuid.uuid4().hex[:10]
    filename  = f"{uid}{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        t0             = time.time()
        class_key, confidence, all_probs = predict_image(str(save_path))
        elapsed         = round((time.time() - t0) * 1000, 1)  # ms
        info           = CLASS_INFO[class_key]

        sorted_probs = sorted(
            [{"class_key": k, "label": CLASS_INFO[k]["label"], "probability": v}
             for k, v in all_probs.items()],
            key=lambda x: x["probability"], reverse=True
        )

        record = {
            "id":           uid,
            "filename":     filename,
            "original":     secure_filename(file.filename),
            "patient_name": patient_name, # Added to record
            "patient_age":  patient_age,  # Added to record
            "class_key":    class_key,
            "label":        info["label"],
            "status":       info["status"],
            "confidence":   confidence,
            "timestamp":    datetime.now().isoformat(),
            "elapsed_ms":   elapsed
        }
        
        # Unga original database logic
        db_insert(record) 

        return jsonify({
            "success":      True,
            "patient_name": patient_name, # Return to frontend
            "patient_age":  patient_age,  # Return to frontend
            "prediction":   class_key,
            "label":        info["label"],
            "status":       info["status"],
            "icon":         info["icon"],
            "color":        info["color"],
            "confidence":   confidence,
            "description":  info["description"],
            "precautions":  info["precautions"],
            "foods":        info.get("foods", []),
            "all_probs":    sorted_probs,
            "elapsed_ms":   elapsed,
            "image_url":    f"/uploads/{filename}",
            "record_id":    uid
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history():
    data = db_load()
    return jsonify(data["predictions"][:20])

@app.route('/stats', methods=['GET'])
def get_stats():
    # Mukkiyamaana fix: Prediction.query.all() thavaru, JSON DB load pannanum
    data = db_load()
    results = data.get("predictions", [])
    
    total = len(results)
    # Healthy cases count
    normal_count = len([r for r in results if r.get("status") == 'HEALTHY'])
    # Abnormal cases (Normal thavira matha ellam)
    abnormal_count = total - normal_count
    
    # Ovvoru disease class-kum unique keys-a map panrom
    class_counts = {
        'cataract': 0,
        'diabetic_retinopathy': 0,
        'glaucoma': 0,
        'macular_degeneration': 0,
        'normal': 0
    }
    
    for r in results:
        # JSON-la irukira class_key-ai direct-ah count panrom
        label_key = r.get("class_key", "").lower()
        if label_key in class_counts:
            class_counts[label_key] += 1
            
    return jsonify({
        'total': total,
        'abnormal': abnormal_count,
        'normal': normal_count,
        'class_counts': class_counts
    })

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Retinal Disease Detection System")
    print(f"  Running at: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)