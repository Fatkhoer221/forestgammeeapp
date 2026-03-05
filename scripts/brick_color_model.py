# scripts/brick_color_model.py
# MYCOTWIN-GUARDIAN — Deteksi ID Brick dengan K-Means Clustering
# Membaca "sidik jari warna" nano-silika dari setiap brick
#
# CARA KERJA:
# 1. Foto brick masuk
# 2. K-Means cari 3 warna dominan di brick
# 3. Warna dominan = ID unik brick
# 4. Cocokkan dengan database → verified/tidak
#
# Jalankan registrasi: python scripts/brick_color_model.py register foto_brick.jpg BRICK_001
# Jalankan deteksi:    python scripts/brick_color_model.py detect foto_brick.jpg

import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import json
import os
import sys
from datetime import datetime

# ================================================
# FILE DATABASE BRICK
# ================================================
BRICK_DB_FILE = 'data/brick_color_database.json'

def load_brick_db():
    os.makedirs('data', exist_ok=True)
    if os.path.exists(BRICK_DB_FILE):
        with open(BRICK_DB_FILE, 'r') as f:
            return json.load(f)
    return {"bricks": {}}

def save_brick_db(db):
    os.makedirs('data', exist_ok=True)
    with open(BRICK_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

# ================================================
# FUNGSI UTAMA: Ekstrak Warna Dominan dengan K-Means
# ================================================
def extract_dominant_colors(image_path, n_colors=3):
    """
    Mengekstrak N warna dominan dari foto brick.
    
    Cara kerja K-Means:
    - Gambar dipecah jadi ribuan pixel
    - K-Means mengelompokkan pixel yang warnanya mirip
    - Hasil = N pusat kelompok = N warna dominan
    
    Returns:
        list: [[R,G,B], [R,G,B], ...] warna dominan urut dari terbanyak
    """
    if not os.path.exists(image_path):
        return None, f"File tidak ditemukan: {image_path}"

    # Load dan resize gambar (224x224 cukup)
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Fokus area TENGAH gambar (bagian nano-silika terlihat)
    h, w = img_array.shape[:2]
    margin = 30
    roi = img_array[margin:h-margin, margin:w-margin]

    # Reshape jadi daftar pixel [R, G, B]
    pixels = roi.reshape(-1, 3).astype(float)

    # Jalankan K-Means
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Hitung proporsi tiap warna
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    # Urutkan dari warna terbanyak
    sorted_idx = np.argsort(counts)[::-1]
    dominant_colors = kmeans.cluster_centers_[sorted_idx].astype(int).tolist()
    proportions = (counts[sorted_idx] / len(labels) * 100).tolist()

    return dominant_colors, proportions


def colors_to_signature(dominant_colors):
    """
    Ubah warna dominan menjadi 'tanda tangan' unik brick.
    Format: "R1G1B1-R2G2B2-R3G3B3"
    """
    parts = []
    for color in dominant_colors:
        parts.append(f"{color[0]:03d}{color[1]:03d}{color[2]:03d}")
    return "-".join(parts)


def color_distance(colors1, colors2):
    """
    Hitung kemiripan antara dua set warna dominan.
    Semakin kecil = semakin mirip = kemungkinan brick sama.
    """
    total_dist = 0
    for c1, c2 in zip(colors1, colors2):
        dist = np.sqrt(sum((a-b)**2 for a,b in zip(c1,c2)))
        total_dist += dist
    return total_dist / len(colors1)


# ================================================
# REGISTRASI BRICK BARU
# ================================================
def register_brick(image_path, brick_id, user_id="admin"):
    """
    Daftarkan brick baru ke database berdasarkan foto.
    
    Args:
        image_path: path foto brick
        brick_id: ID yang ingin diberikan (misal "BRICK_001")
        user_id: siapa yang mendaftarkan
    """
    print(f"\n{'='*55}")
    print(f"  MYCOTWIN — Registrasi Brick: {brick_id}")
    print(f"{'='*55}")

    db = load_brick_db()

    if brick_id in db["bricks"]:
        print(f"⚠️  Brick {brick_id} sudah terdaftar!")
        print(f"   Terdaftar oleh: {db['bricks'][brick_id]['registered_by']}")
        return False

    # Ekstrak warna dominan
    print(f"\n📸 Menganalisis foto: {image_path}")
    dominant_colors, proportions = extract_dominant_colors(image_path, n_colors=3)

    if dominant_colors is None:
        print(f"❌ Error: {proportions}")
        return False

    # Tampilkan warna yang terdeteksi
    print(f"\n🎨 Warna Dominan Terdeteksi:")
    color_names = []
    for i, (color, prop) in enumerate(zip(dominant_colors, proportions)):
        r, g, b = color
        name = rgb_to_name(r, g, b)
        color_names.append(name)
        bar = "█" * int(prop / 5)
        print(f"   Warna {i+1}: RGB({r:3d},{g:3d},{b:3d}) = {name:10s} {bar} {prop:.1f}%")

    # Simpan ke database
    signature = colors_to_signature(dominant_colors)
    db["bricks"][brick_id] = {
        "brick_id": brick_id,
        "dominant_colors": dominant_colors,
        "color_names": color_names,
        "proportions": [round(p, 2) for p in proportions],
        "signature": signature,
        "registered_by": user_id,
        "registered_at": datetime.now().isoformat(),
        "image_path": image_path
    }
    save_brick_db(db)

    print(f"\n✅ Brick {brick_id} berhasil didaftarkan!")
    print(f"   Sidik jari warna: {signature}")
    print(f"   Didaftarkan oleh: {user_id}")
    print(f"{'='*55}")
    return True


# ================================================
# DETEKSI & VERIFIKASI BRICK
# ================================================
def detect_brick(image_path, threshold=80):
    """
    Deteksi brick mana yang ada di foto berdasarkan warna.
    
    Args:
        image_path: path foto yang akan dideteksi
        threshold: batas jarak warna (makin kecil = makin ketat)
    
    Returns:
        dict: hasil deteksi
    """
    db = load_brick_db()

    if not db["bricks"]:
        return {
            "verified": False,
            "brick_id": None,
            "message": "❌ Database brick kosong! Daftarkan brick dulu."
        }

    # Ekstrak warna dari foto baru
    dominant_colors, proportions = extract_dominant_colors(image_path, n_colors=3)

    if dominant_colors is None:
        return {"verified": False, "message": f"❌ Error: {proportions}"}

    # Cocokkan dengan semua brick di database
    best_match_id = None
    best_distance = float('inf')

    for brick_id, brick_data in db["bricks"].items():
        db_colors = brick_data["dominant_colors"]
        distance = color_distance(dominant_colors, db_colors)

        if distance < best_distance:
            best_distance = distance
            best_match_id = brick_id

    # Hitung confidence (semakin dekat = makin yakin)
    confidence = max(0, 1 - (best_distance / threshold))

    if best_distance <= threshold and confidence > 0.3:
        brick_data = db["bricks"][best_match_id]
        return {
            "verified": True,
            "brick_id": best_match_id,
            "confidence": round(confidence, 3),
            "distance": round(best_distance, 2),
            "color_names": brick_data["color_names"],
            "registered_by": brick_data["registered_by"],
            "message": f"✅ Brick terverifikasi! ID: {best_match_id} (confidence: {confidence*100:.1f}%)"
        }
    else:
        return {
            "verified": False,
            "brick_id": None,
            "confidence": 0.0,
            "distance": round(best_distance, 2),
            "best_guess": best_match_id,
            "message": f"❌ Brick tidak dikenali! Jarak warna terlalu jauh ({best_distance:.1f})"
        }


# ================================================
# HELPER: Tebak nama warna dari RGB
# ================================================
def rgb_to_name(r, g, b):
    """Konversi RGB ke nama warna sederhana"""
    if r > 150 and g < 100 and b < 100:
        return "Merah"
    elif r > 150 and g > 100 and b < 80:
        return "Oranye"
    elif r > 150 and g > 150 and b < 80:
        return "Kuning"
    elif r < 100 and g > 120 and b < 100:
        return "Hijau"
    elif r < 80 and g < 80 and b > 150:
        return "Biru"
    elif r > 100 and g < 80 and b > 100:
        return "Ungu"
    elif r > 150 and g < 80 and b > 100:
        return "Merah Muda"
    elif r > 150 and g > 150 and b > 150:
        return "Putih"
    elif r < 80 and g < 80 and b < 80:
        return "Hitam"
    elif abs(r-g) < 30 and abs(g-b) < 30:
        return "Abu-abu"
    elif r > 120 and g > 80 and b < 60:
        return "Coklat"
    else:
        return f"RGB({r},{g},{b})"


# ================================================
# SIMULASI DATABASE (untuk demo tanpa brick asli)
# ================================================
def create_demo_database():
    """
    Buat database demo dengan warna-warna berbeda
    untuk simulasi sebelum brick asli jadi.
    """
    print("\n🎭 Membuat database demo...")
    db = {"bricks": {}}

    demo_bricks = {
        "MTG-001": [[180, 50, 50], [200, 80, 60], [160, 40, 40]],   # Merah
        "MTG-002": [[50, 150, 50], [60, 180, 70], [40, 130, 40]],   # Hijau
        "MTG-003": [[50, 80, 180], [60, 90, 200], [40, 70, 160]],   # Biru
        "MTG-004": [[180, 160, 50], [200, 180, 60], [160, 140, 40]], # Kuning
        "MTG-005": [[150, 50, 150], [170, 60, 170], [130, 40, 130]], # Ungu
    }

    for brick_id, colors in demo_bricks.items():
        color_names = [rgb_to_name(*c) for c in colors]
        db["bricks"][brick_id] = {
            "brick_id": brick_id,
            "dominant_colors": colors,
            "color_names": color_names,
            "proportions": [40.0, 35.0, 25.0],
            "signature": colors_to_signature(colors),
            "registered_by": "demo",
            "registered_at": datetime.now().isoformat(),
            "image_path": "demo"
        }
        print(f"   ✅ {brick_id} → {' + '.join(color_names)}")

    save_brick_db(db)
    print(f"\n✅ Database demo siap! {len(demo_bricks)} brick terdaftar.")
    print(f"   Tersimpan di: {BRICK_DB_FILE}")


# ================================================
# JALANKAN LANGSUNG
# ================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════╗
║  MYCOTWIN — Brick Color Detection System     ║
╠══════════════════════════════════════════════╣
║  Cara pakai:                                 ║
║                                              ║
║  1. Buat demo database:                      ║
║     python scripts/brick_color_model.py demo ║
║                                              ║
║  2. Daftarkan brick baru:                    ║
║     python scripts/brick_color_model.py      ║
║     register foto.jpg MTG-001 user123        ║
║                                              ║
║  3. Deteksi brick dari foto:                 ║
║     python scripts/brick_color_model.py      ║
║     detect foto.jpg                          ║
╚══════════════════════════════════════════════╝
        """)
        sys.exit(0)

    command = sys.argv[1]

    if command == "demo":
        create_demo_database()

    elif command == "register":
        if len(sys.argv) < 4:
            print("Cara: python scripts/brick_color_model.py register foto.jpg MTG-001")
        else:
            img = sys.argv[2]
            bid = sys.argv[3]
            uid = sys.argv[4] if len(sys.argv) > 4 else "user"
            register_brick(img, bid, uid)

    elif command == "detect":
        if len(sys.argv) < 3:
            print("Cara: python scripts/brick_color_model.py detect foto.jpg")
        else:
            img = sys.argv[2]
            print(f"\n{'='*55}")
            print(f"  MYCOTWIN — Deteksi Brick dari Foto")
            print(f"{'='*55}")
            result = detect_brick(img)
            print(f"\n  {result['message']}")
            if result.get('brick_id'):
                print(f"  Brick ID   : {result['brick_id']}")
                print(f"  Confidence : {result['confidence']*100:.1f}%")
                print(f"  Warna      : {result.get('color_names', [])}")
            print(f"{'='*55}")

    elif command == "list":
        db = load_brick_db()
        print(f"\n📋 Daftar Brick Terdaftar ({len(db['bricks'])} brick):")
        for bid, data in db["bricks"].items():
            print(f"  {bid} → {' + '.join(data['color_names'])} | by {data['registered_by']}")