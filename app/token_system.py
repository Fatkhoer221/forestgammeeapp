# app/token_system.py
# MYCOTWIN-GUARDIAN — Sistem Token Reward

import json
import os
from datetime import datetime

TOKEN_DB_FILE = 'data/token_database.json'

def load_token_db():
    os.makedirs('data', exist_ok=True)
    if os.path.exists(TOKEN_DB_FILE):
        with open(TOKEN_DB_FILE, 'r') as f:
            return json.load(f)
    return {"users": {}, "transactions": []}

def save_token_db(db):
    os.makedirs('data', exist_ok=True)
    with open(TOKEN_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def calculate_token(classification, confidence, brick_verified=True):
    """
    Hitung token berdasarkan hasil AI.
    brick_verified hanya wajib kalau database brick sudah ada isinya.
    """
    # Cek apakah database brick sudah ada isinya
    brick_db_path = 'data/brick_color_database.json'
    db_has_bricks = False
    if os.path.exists(brick_db_path):
        with open(brick_db_path) as f:
            db = json.load(f)
            db_has_bricks = len(db.get("bricks", {})) > 0

    # Kalau database brick sudah ada → brick harus terverifikasi
    if db_has_bricks and not brick_verified:
        return {
            "token_earned": 0,
            "base_token": 0,
            "bonus_token": 0,
            "classification": classification,
            "confidence": round(confidence, 3),
            "reason": "Brick tidak terverifikasi — daftarkan brick asli dulu!"
        }

    # Hitung token berdasarkan klasifikasi
    if classification != "alive":
        return {
            "token_earned": 0,
            "base_token": 0,
            "bonus_token": 0,
            "classification": classification,
            "confidence": round(confidence, 3),
            "reason": "Tanaman tidak hidup — 0 token"
        }

    # Base token untuk tanaman hidup
    base_token = 50

    # Bonus confidence
    bonus = 0
    if confidence >= 0.90:
        bonus = 10
    elif confidence >= 0.80:
        bonus = 5
    elif confidence >= 0.70:
        bonus = 2

    total = base_token + bonus

    return {
        "token_earned": total,
        "base_token": base_token,
        "bonus_token": bonus,
        "classification": classification,
        "confidence": round(confidence, 3),
        "reason": f"Tanaman hidup! Base +{base_token} | Bonus confidence +{bonus}"
    }

def grant_token(user_id, brick_id, token_result, image_path=""):
    """Catat token ke database"""
    db = load_token_db()

    if user_id not in db["users"]:
        db["users"][user_id] = {
            "user_id": user_id,
            "total_token": 0,
            "total_submissions": 0,
            "created_at": datetime.now().isoformat()
        }

    earned = token_result.get("token_earned", 0)
    db["users"][user_id]["total_token"] += earned
    db["users"][user_id]["total_submissions"] += 1

    transaction = {
        "transaction_id": f"TX{len(db['transactions'])+1:06d}",
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "brick_id": brick_id,
        "token_earned": earned,
        "classification": token_result.get("classification"),
        "confidence": token_result.get("confidence"),
        "image": os.path.basename(image_path) if image_path else ""
    }
    db["transactions"].append(transaction)
    save_token_db(db)

    total = db["users"][user_id]["total_token"]
    return {
        "success": True,
        "transaction_id": transaction["transaction_id"],
        "token_earned": earned,
        "total_token": total,
        "message": f"+{earned} token | Total: {total} token"
    }

def get_balance(user_id):
    db = load_token_db()
    if user_id in db["users"]:
        user = db["users"][user_id]
        return {
            "user_id": user_id,
            "total_token": user["total_token"],
            "total_submissions": user["total_submissions"],
            "message": f"Saldo: {user['total_token']} token dari {user['total_submissions']} verifikasi"
        }
    return {
        "user_id": user_id,
        "total_token": 0,
        "total_submissions": 0,
        "message": "User belum pernah submit"
    }

def get_leaderboard(top_n=10):
    db = load_token_db()
    users = list(db["users"].values())
    users.sort(key=lambda x: x["total_token"], reverse=True)
    return users[:top_n]


if __name__ == '__main__':
    print("=" * 50)
    print("  MYCOTWIN — Test Token System")
    print("=" * 50)

    test_cases = [
        ("alive",    0.94, True,  "user_001", "BRICK_001"),
        ("alive",    0.75, True,  "user_001", "BRICK_002"),
        ("dead",     0.88, True,  "user_002", "BRICK_003"),
        ("no_plant", 0.91, True,  "user_003", "BRICK_004"),
        ("alive",    0.93, False, "user_001", "BRICK_PALSU"),
    ]

    for cls, conf, verified, uid, bid in test_cases:
        token_result = calculate_token(cls, conf, verified)
        grant_result = grant_token(uid, bid, token_result)
        print(f"\n  User: {uid} | Brick: {bid}")
        print(f"  AI: {cls.upper()} ({conf*100:.0f}%)")
        print(f"  {grant_result['message']}")

    print("\n" + "="*50)
    print("  LEADERBOARD:")
    for i, u in enumerate(get_leaderboard(), 1):
        print(f"  #{i} {u['user_id']}: {u['total_token']} token")
    print("="*50)