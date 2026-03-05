# scripts/plant_detector_yolo.py
# MYCOTWIN-GUARDIAN — Step 2: Deteksi Tanaman dengan YOLOv8
# Install dulu: pip install ultralytics
# Jalankan test: python scripts/plant_detector_yolo.py foto_brick.jpg

import os
import sys
import numpy as np

def check_ultralytics():
    """Cek apakah ultralytics (YOLO) sudah terinstall"""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False

def detect_plant_yolo(image_path, confidence_threshold=0.3):
    """
    Mendeteksi keberadaan tanaman dalam gambar menggunakan YOLOv8.
    
    Menggunakan model YOLOv8 pretrained yang sudah bisa mendeteksi
    tanaman (class 'potted plant' dan 'plant' dari COCO dataset).
    
    Returns:
        dict: {
            "plant_detected": bool,
            "confidence": float,
            "count": int,
            "bounding_boxes": list,
            "message": str
        }
    """
    if not os.path.exists(image_path):
        return {"plant_detected": False, "message": f"File tidak ditemukan"}

    if not check_ultralytics():
        # Fallback ke OpenCV jika YOLO belum terinstall
        print("⚠️  ultralytics belum terinstall, pakai fallback OpenCV...")
        return detect_plant_opencv_fallback(image_path)

    from ultralytics import YOLO

    # Download model YOLOv8n otomatis pertama kali (sekitar 6MB)
    model_path = 'models/yolov8n.pt'
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    model = YOLO('yolov8n.pt')  # nano = paling cepat, cukup untuk deteksi

    # Jalankan deteksi
    results = model(image_path, verbose=False, conf=confidence_threshold)

    # COCO class IDs yang relevan:
    # 58 = potted plant, 63 = dining table (sering false positive), dll
    PLANT_CLASS_IDS = {58: "potted plant", 60: "dining table"}
    # Untuk proyek ini, kita lebih fleksibel - deteksi semua objek organik

    plant_detected = False
    best_confidence = 0.0
    boxes = []
    count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Deteksi tanaman (class 58 = potted plant di COCO)
            if class_id == 58:
                plant_detected = True
                count += 1
                if conf > best_confidence:
                    best_confidence = conf
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append({
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                    "confidence": round(conf, 3)
                })

    if plant_detected:
        return {
            "plant_detected": True,
            "confidence": round(best_confidence, 3),
            "count": count,
            "bounding_boxes": boxes,
            "message": f"✅ Terdeteksi {count} tanaman (confidence: {best_confidence*100:.1f}%)"
        }
    else:
        # Jika YOLO tidak deteksi, kita tetap lanjut ke CNN classifier
        # (CNN lebih handal untuk kasus spesifik brick)
        return {
            "plant_detected": None,  # None = tidak pasti, lanjut ke CNN
            "confidence": 0.0,
            "count": 0,
            "bounding_boxes": [],
            "message": "⚠️  YOLO tidak mendeteksi tanaman pot. Lanjut ke CNN classifier..."
        }


def detect_plant_opencv_fallback(image_path):
    """
    Fallback: Deteksi tanaman menggunakan analisis warna hijau (OpenCV)
    Digunakan jika YOLO belum terinstall
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return {"plant_detected": False, "message": "Gagal membaca gambar"}

    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Deteksi warna hijau (tanaman hidup)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Deteksi warna coklat (tanaman mati/substrat)
    lower_brown = np.array([10, 40, 20])
    upper_brown = np.array([20, 200, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    total_pixels = 224 * 224
    green_ratio = np.sum(mask_green > 0) / total_pixels
    brown_ratio = np.sum(mask_brown > 0) / total_pixels

    if green_ratio > 0.05:  # Lebih dari 5% pixel hijau
        return {
            "plant_detected": True,
            "confidence": round(min(green_ratio * 3, 0.95), 3),
            "count": 1,
            "bounding_boxes": [],
            "method": "opencv_color",
            "message": f"✅ Tanaman terdeteksi via warna hijau ({green_ratio*100:.1f}% area)"
        }
    else:
        return {
            "plant_detected": None,
            "confidence": 0.0,
            "count": 0,
            "bounding_boxes": [],
            "method": "opencv_color",
            "message": "⚠️  Warna hijau minim. Lanjut ke CNN classifier..."
        }


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print("\n" + "="*50)
        print("  MYCOTWIN — Deteksi Tanaman YOLO (Step 2)")
        print("="*50)
        result = detect_plant_yolo(img_path)
        print(f"\n  {result['message']}")
        if result.get('bounding_boxes'):
            print(f"  Jumlah terdeteksi : {result['count']}")
            print(f"  Confidence        : {result['confidence']*100:.1f}%")
        print("="*50)
    else:
        print("Cara pakai: python scripts/plant_detector_yolo.py foto_brick.jpg")
        print("\nPastikan ultralytics terinstall:")
        print("  pip install ultralytics")
