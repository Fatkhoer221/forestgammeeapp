# 🍄 MYCOTWIN-GUARDIAN — Panduan Lengkap

## Struktur File

```
mycotwin-guardian/
├── dataset/
│   ├── train/
│   │   ├── alive/       ← foto tanaman HIDUP (min 30 foto)
│   │   ├── dead/        ← foto tanaman MATI (min 30 foto)
│   │   └── no_plant/    ← foto area KOSONG (min 30 foto)
│   └── validation/
│       ├── alive/       ← 10 foto berbeda
│       ├── dead/
│       └── no_plant/
├── models/              ← model AI tersimpan di sini (otomatis)
├── data/                ← database token (otomatis)
├── scripts/
│   ├── augment_dataset.py      ← perbanyak foto otomatis
│   ├── train_model.py          ← training CNN MobileNetV2
│   ├── brick_id_detector.py    ← Step 1: deteksi nano-silika
│   ├── plant_detector_yolo.py  ← Step 2: deteksi YOLO
│   └── test_api.py             ← test semua endpoint
├── app/
│   ├── app.py                  ← Flask API utama
│   └── token_system.py         ← sistem token reward
└── venv/                       ← virtual environment Python
```

## Urutan Menjalankan

### 1. Siapkan Dataset
Foto pakai HP, minimal 30 foto per kelas, masukkan ke folder:
- `dataset/train/alive/` → tanaman segar/hidup
- `dataset/train/dead/` → tanaman layu/mati
- `dataset/train/no_plant/` → pot/substrat kosong

### 2. Augmentasi (Perbanyak Foto Otomatis)
```cmd
python scripts/augment_dataset.py
```

### 3. Training Model AI
```cmd
python scripts/train_model.py
```
Tunggu 20-30 menit. Model tersimpan di `models/`

### 4. Install YOLO (opsional)
```cmd
pip install ultralytics
```

### 5. Jalankan API
```cmd
python app/app.py
```
API berjalan di http://localhost:5000

### 6. Test API
```cmd
python scripts/test_api.py path/foto_brick.jpg
```

## Endpoint API

| Method | URL | Fungsi |
|--------|-----|--------|
| GET | / | Status server |
| POST | /verify | Pipeline lengkap + token |
| POST | /classify | Klasifikasi saja |
| GET | /balance/{user_id} | Cek saldo token |
| GET | /leaderboard | Top pengguna |

## Pipeline Verifikasi

```
Foto Brick → Step 1 (Nano-Silika ID) → Step 2 (YOLO) → Step 3 (CNN)
                                                              ↓
                                          Semua OK? → AUTO-GRANT TOKEN 🪙
                                          Ada gagal? → TOLAK + FEEDBACK
```
