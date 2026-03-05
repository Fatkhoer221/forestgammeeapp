# scripts/augment_dataset.py
# MYCOTWIN-GUARDIAN — Perbanyak foto dataset otomatis
# Jalankan: python scripts/augment_dataset.py

import tensorflow as tf
import numpy as np
from PIL import Image
import os

def augment_images(source_folder, target_count=150):
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ])

    for class_name in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)
        print(f"\nKelas '{class_name}': {current_count} gambar asli")

        if current_count == 0:
            print(f"  ⚠️  Folder kosong! Isi dulu dengan foto.")
            continue

        aug_count = 0
        i = 0
        while current_count + aug_count < target_count:
            img_path = os.path.join(class_path, images[i % len(images)])
            try:
                img = Image.open(img_path).convert('RGB').resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = tf.expand_dims(img_array, 0)
                augmented = augment(img_tensor, training=True)
                aug_img = Image.fromarray(
                    (augmented[0].numpy() * 255).astype(np.uint8)
                )
                save_path = os.path.join(class_path, f'aug_{aug_count:04d}.jpg')
                aug_img.save(save_path)
                aug_count += 1
            except Exception as e:
                print(f"  Error: {e}")
            i += 1

        print(f"  ✅ Total sekarang: {current_count + aug_count} gambar")

if __name__ == '__main__':
    print("=" * 50)
    print("  MYCOTWIN-GUARDIAN — Augmentasi Dataset")
    print("=" * 50)

    print("\n📂 Augmentasi folder TRAIN...")
    augment_images('dataset/train', target_count=150)

    print("\n📂 Augmentasi folder VALIDATION...")
    augment_images('dataset/validation', target_count=50)

    print("\n✅ Augmentasi selesai! Dataset siap untuk training.")
