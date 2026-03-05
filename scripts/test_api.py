# scripts/test_api.py
# MYCOTWIN-GUARDIAN — Test semua endpoint API
# Pastikan app.py sudah jalan dulu di terminal lain:
#   python app/app.py
# Lalu jalankan: python scripts/test_api.py foto_brick.jpg

import requests
import sys
import json

BASE_URL = "http://localhost:5000"

def test_health():
    print("\n[1] Test: Health Check")
    r = requests.get(f"{BASE_URL}/")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

def test_classify(image_path):
    print("\n[2] Test: Klasifikasi CNN")
    with open(image_path, 'rb') as f:
        r = requests.post(f"{BASE_URL}/classify", files={"image": f})
    data = r.json()
    print(f"  Hasil    : {data.get('label')}")
    print(f"  Confidence: {data.get('confidence_pct')}")
    print(f"  Semua    : {data.get('all_probabilities')}")

def test_verify(image_path, user_id="test_user"):
    print(f"\n[3] Test: Pipeline Lengkap (user: {user_id})")
    with open(image_path, 'rb') as f:
        r = requests.post(
            f"{BASE_URL}/verify",
            files={"image": f},
            data={"user_id": user_id}
        )
    data = r.json()
    print(f"  Outcome  : {data.get('outcome')}")
    print(f"  Feedback : {data.get('feedback')}")
    if data.get('token'):
        print(f"  Token    : {data['token'].get('message','')}")

def test_balance(user_id):
    print(f"\n[4] Test: Cek Saldo (user: {user_id})")
    r = requests.get(f"{BASE_URL}/balance/{user_id}")
    print(f"  {r.json().get('message')}")

def test_leaderboard():
    print("\n[5] Test: Leaderboard")
    r = requests.get(f"{BASE_URL}/leaderboard")
    data = r.json()
    for i, u in enumerate(data.get('leaderboard', []), 1):
        print(f"  #{i} {u['user_id']}: {u['total_token']} token")

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 50)
    print("  MYCOTWIN — API Test Suite")
    print("=" * 50)

    try:
        test_health()

        if image_path:
            test_classify(image_path)
            test_verify(image_path, "petani_001")
            test_balance("petani_001")
            test_leaderboard()
        else:
            print("\n  ⚠️  Untuk test penuh, sediakan path foto:")
            print("  python scripts/test_api.py path/foto.jpg")

        print("\n✅ Semua test selesai!")
    except requests.exceptions.ConnectionError:
        print("\n❌ Tidak bisa connect ke server!")
        print("   Pastikan app.py sudah jalan:")
        print("   python app/app.py")
