# app/app.py
# MYCOTWIN-GUARDIAN — ForestGem-App Backend API
# Pipeline: Brick ID → YOLO → Roboflow CNN → Token

import os
import sys
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS          # ← TAMBAHAN untuk Railway/Vercel
from PIL import Image
import io
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)                            # ← TAMBAHAN agar Vercel bisa konek ke Railway

# ================================================
# ENDPOINT: Serve Frontend
# ================================================
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# ================================================
# ENDPOINT UTAMA: Pipeline Verifikasi Lengkap
# POST /verify
# ================================================
@app.route('/verify', methods=['POST'])
def verify_pipeline():
    user_id = request.form.get('user_id', 'anonymous')

    if 'image' not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file gambar!"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    temp_path = f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.jpg'
    with open(temp_path, 'wb') as f:
        f.write(image_bytes)

    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "steps": {}
    }

    try:
        # ================================================
        # STEP 1: Deteksi Brick (K-Means Nano-Silika)
        # ================================================
        try:
            from scripts.brick_color_model import detect_brick
            step1 = detect_brick(temp_path)
        except Exception as e:
            step1 = {
                "verified": False,
                "brick_id": None,
                "message": f"Brick detector error: {str(e)}"
            }
        result["steps"]["step1_brick_id"] = step1
        brick_ok = step1.get("verified", False)
        brick_id = step1.get("brick_id", "UNKNOWN")

        # ================================================
        # STEP 2: Deteksi Tanaman (YOLO)
        # ================================================
        try:
            from scripts.plant_detector_yolo import detect_plant_yolo
            step2 = detect_plant_yolo(temp_path)
        except Exception as e:
            step2 = {
                "plant_detected": None,
                "confidence": 0.0,
                "count": 0,
                "message": f"YOLO error: {str(e)}"
            }
        result["steps"]["step2_yolo_detection"] = step2
        plant_ok = (step2.get("plant_detected") != False)

        # ================================================
        # STEP 3: Klasifikasi (Roboflow ViT - 96.9%!)
        # ================================================
        try:
            from scripts.classify_roboflow import classify_image_roboflow
            step3 = classify_image_roboflow(temp_path)
        except Exception as e:
            step3 = {
                "classification": "no_plant",
                "confidence": 0.0,
                "all_probabilities": {},
                "source": "error",
                "error": str(e)
            }
        result["steps"]["step3_cnn"] = step3
        classification = step3.get("classification", "no_plant")
        confidence = step3.get("confidence", 0.0)
        cnn_ok = (classification == "alive")

        # ================================================
        # KEPUTUSAN TOKEN
        # ================================================
        from app.token_system import calculate_token, grant_token

        token_calc = calculate_token(
            classification=classification,
            confidence=confidence,
            brick_verified=brick_ok
        )

        # Kriteria dapat token:
        # brick_ok → wajib setelah brick asli datang
        # Sementara (brick belum ada): cukup tanaman hidup
        brick_db_exists = os.path.exists('data/brick_color_database.json')
        db_has_bricks = False
        if brick_db_exists:
            with open('data/brick_color_database.json') as f:
                db = json.load(f)
                db_has_bricks = len(db.get("bricks", {})) > 0

        if db_has_bricks:
            approved = brick_ok and plant_ok and cnn_ok
        else:
            approved = plant_ok and cnn_ok

        if approved:
            token_grant = grant_token(user_id, brick_id, token_calc, temp_path)
            result["outcome"] = "APPROVED"
            result["token"] = token_grant
            result["feedback"] = f"Verifikasi berhasil! {token_grant.get('message','')}"
        else:
            result["outcome"] = "REJECTED"
            result["token"] = {"token_earned": 0}

            reasons = []
            if db_has_bricks and not brick_ok:
                reasons.append("ID Brick tidak terverifikasi")
            if classification == "dead":
                reasons.append("Tanaman terdeteksi mati — rawat tanamanmu!")
            if classification == "no_plant":
                reasons.append("Tidak ada tanaman — pastikan tanaman terlihat jelas")
            if not plant_ok:
                reasons.append("Tanaman tidak terdeteksi oleh YOLO")

            result["feedback"] = " | ".join(reasons) if reasons else "Kriteria tidak terpenuhi"

        result["success"] = True
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "user_id": user_id
        }), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ================================================
# ENDPOINT: Klasifikasi Saja
# POST /classify
# ================================================
@app.route('/classify', methods=['POST'])
def classify_only():
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar!"}), 400

    image_bytes = request.files['image'].read()
    temp_path = f'temp_cls_{datetime.now().strftime("%H%M%S%f")}.jpg'

    with open(temp_path, 'wb') as f:
        f.write(image_bytes)

    try:
        from scripts.classify_roboflow import classify_image_roboflow
        result = classify_image_roboflow(temp_path)

        EMOJI = {"alive": "🌱", "dead": "💀", "no_plant": "📦"}
        cls = result.get("classification", "unknown")
        conf = result.get("confidence", 0.0)

        return jsonify({
            "classification": cls,
            "label": f"{EMOJI.get(cls,'')} {cls.upper()}",
            "confidence": conf,
            "confidence_pct": f"{conf*100:.1f}%",
            "all_probabilities": result.get("all_probabilities", {}),
            "source": result.get("source", "unknown")
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ================================================
# ENDPOINT: Saldo Token
# GET /balance/<user_id>
# ================================================
@app.route('/balance/<user_id>', methods=['GET'])
def check_balance(user_id):
    from app.token_system import get_balance
    return jsonify(get_balance(user_id))


# ================================================
# ENDPOINT: Leaderboard
# GET /leaderboard
# ================================================
@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    from app.token_system import get_leaderboard
    top = get_leaderboard(10)
    return jsonify({
        "leaderboard": top,
        "total_users": len(top)
    })


# ================================================
# JALANKAN SERVER — support Railway (PORT env)
# ================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  MYCOTWIN-GUARDIAN ForestGem-App API")
    print("  Roboflow ViT Model - Akurasi 96.9%!")
    print("=" * 55)
    port = int(os.environ.get('PORT', 5000))      # ← DIUBAH: baca PORT dari Railway
    print(f"\n  Server: http://localhost:{port}")
    print("  Ctrl+C untuk stop\n")
    app.run(debug=False, host='0.0.0.0', port=port)  # ← DIUBAH: debug=False untuk production