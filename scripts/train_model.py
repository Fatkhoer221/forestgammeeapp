# scripts/train_model.py
# MYCOTWIN-GUARDIAN — AI Plant Classification Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# ================================================
# KONFIGURASI
# ================================================
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 30
DATASET_DIR = 'dataset'
MODEL_DIR   = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 55)
print("  MYCOTWIN-GUARDIAN — Training AI Model")
print("=" * 55)

# ================================================
# LOAD DATASET
# ================================================
print("\n📂 Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'validation'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
    seed=42
)

class_names = train_ds.class_names
print(f"✅ Kelas terdeteksi: {class_names}")

with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ================================================
# DATA AUGMENTATION
# ================================================
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
], name='data_augmentation')

# ================================================
# BANGUN MODEL — Transfer Learning MobileNetV2
# ================================================
print("\n🧠 Membangun model MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs, name='mycotwin_classifier')
model.summary()

# ================================================
# COMPILE & TRAINING TAHAP 1
# ================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=3,
        monitor='val_loss',
        verbose=1,
        min_lr=1e-7
    )
]

print("\n🚀 Tahap 1: Training kepala model baru...")
history1 = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=callbacks
)

# ================================================
# FINE-TUNING
# ================================================
print("\n🔧 Tahap 2: Fine-tuning...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    initial_epoch=len(history1.history['accuracy'])
)

# ================================================
# SIMPAN & PLOT
# ================================================
model.save(os.path.join(MODEL_DIR, 'plant_classifier_final.keras'))
print("\n✅ Model disimpan!")

loss, acc = model.evaluate(val_ds)
print(f"\n📊 Validation Accuracy : {acc*100:.2f}%")
print(f"📊 Validation Loss     : {loss:.4f}")

acc_hist   = history1.history['accuracy']  + history2.history['accuracy']
val_hist   = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_hist  = history1.history['loss']  + history2.history['loss']
vloss_hist = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(acc_hist,  label='Train Accuracy',  color='royalblue')
plt.plot(val_hist,  label='Val Accuracy',    color='tomato')
plt.title('Akurasi — MYCOTWIN-GUARDIAN')
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss_hist,  label='Train Loss',  color='royalblue')
plt.plot(vloss_hist, label='Val Loss',    color='tomato')
plt.title('Loss — MYCOTWIN-GUARDIAN')
plt.legend(); plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150)
plt.show()
print("✅ Grafik tersimpan di models/training_history.png")