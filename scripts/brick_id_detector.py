# scripts/brick_id_detector.py
# MYCOTWIN-GUARDIAN — Step 1: Deteksi ID Brick via Nano-Silika Color Recognition
# Setiap brick punya warna nano-silika unik sebagai ID
# Jalankan test: python scripts/brick_id_detector.py foto_brick.jpg

import cv2
import numpy as np
import json
import os
import hashlib
from datetime import datetime

# ================================================
# DATABASE WARNA NANO-SILIKA (ID BRICK)
# Setiap batch brick punya kombinasi warna unik
# Format: { "BRICK_ID": { "hue_min": x, "hue_max": x, ... } }
# ================================================
BRICK_DATABASE = {
    "BRICK_001": {"hue_range": [0, 10],   "name": "Merah", "batch": "Batch-A"},
    "BRICK_002": {"hue_range": [20, 35],  "name": "Kuning","batch": "Batch-A"},
    "BRICK_003": {"hue_range": [36, 85],  "name": "Hijau", "batch": "Batch-B"},
    "BRICK_004": {"hue_range": [86, 130], "name": "Biru",  "batch": "Batch-B"},
    "BRICK_005": {"hue_range": [131,170], "name": "Ungu",  "batch": "Batch-C"},
}

def detect_brick_id(image_path):
    """
    Mendeteksi ID brick berdasarkan warna nano-silika dalam gambar.
    
    Returns:
        dict: {
            "brick_id": str atau None,
            "confidence": float,
            "color_detected": str,
            "verified": bool,
            "message": str
        }
    """
    if not os.path.exists(image_path):
        return {"verified": False, "message": f"File tidak ditemukan: {image_path}"}

    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        return {"verified": False, "message": "Gagal membaca gambar"}

    # Resize untuk konsistensi
    img = cv2.resize(img, (224, 224))

    # Konversi ke HSV (lebih mudah deteksi warna)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Fokus ke area tengah gambar (tempat nano-silika biasanya)
    h, w = hsv.shape[:2]
    roi = hsv[h//4:3*h//4, w//4:3*w//4]  # Region of Interest: tengah

    # Hitung histogram hue
    hue_channel = roi[:, :, 0]
    mean_hue = np.mean(hue_channel)
    saturation = np.mean(roi[:, :, 1])

    # Cek apakah warna cukup jenuh (bukan putih/abu/hitam)
    if saturation < 30:
        return {
            "verified": False,
            "brick_id": None,
            "confidence": 0.0,
            "color_detected": "Tidak terdeteksi (warna terlalu pucat)",
            "message": "❌ Nano-silika tidak terdeteksi. Pastikan brick terlihat jelas."
        }

    # Cocokkan dengan database
    detected_id = None
    detected_name = "Unknown"
    best_confidence = 0.0

    for brick_id, data in BRICK_DATABASE.items():
        h_min, h_max = data["hue_range"]
        if h_min <= mean_hue <= h_max:
            # Confidence berdasarkan seberapa dekat ke tengah range
            range_center = (h_min + h_max) / 2
            distance = abs(mean_hue - range_center)
            range_size = (h_max - h_min) / 2
            confidence = max(0, 1 - (distance / range_size)) * (saturation / 255)

            if confidence > best_confidence:
                best_confidence = confidence
                detected_id = brick_id
                detected_name = data["name"]

    if detected_id and best_confidence > 0.3:
        return {
            "verified": True,
            "brick_id": detected_id,
            "confidence": round(float(best_confidence), 3),
            "color_detected": detected_name,
            "batch": BRICK_DATABASE[detected_id]["batch"],
            "message": f"✅ Brick terverifikasi! ID: {detected_id} | Warna: {detected_name}"
        }
    else:
        return {
            "verified": False,
            "brick_id": None,
            "confidence": round(float(best_confidence), 3),
            "color_detected": f"Tidak cocok (Hue: {mean_hue:.1f})",
            "message": "❌ ID Brick tidak dikenali. Kemungkinan bukan brick MYCOTWIN."
        }


def generate_brick_id_hash(image_path):
    """
    Generate hash unik dari gambar sebagai backup ID
    (untuk anti-duplikat foto yang sama dikirim dua kali)
    """
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12].upper()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print("\n" + "="*50)
        print("  MYCOTWIN — Deteksi ID Brick (Step 1)")
        print("="*50)
        result = detect_brick_id(img_path)
        print(f"\n  {result['message']}")
        if result.get('brick_id'):
            print(f"  Brick ID   : {result['brick_id']}")
            print(f"  Batch      : {result.get('batch','?')}")
            print(f"  Confidence : {result['confidence']*100:.1f}%")
            print(f"  Warna      : {result['color_detected']}")
        print("="*50)
    else:
        print("Cara pakai: python scripts/brick_id_detector.py foto_brick.jpg")
