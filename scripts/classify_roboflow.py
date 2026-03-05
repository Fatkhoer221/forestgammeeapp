import requests
import base64
import json
from PIL import Image
import io
import os

ROBOFLOW_API_KEY = "VVQ4LFOI6dtVb5YAJ3a4"
ROBOFLOW_MODEL_ID = "mycotwin-guardian/1"

def classify_image_roboflow(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        url = f"https://classify.roboflow.com/{ROBOFLOW_MODEL_ID}"
        response = requests.post(
            url,
            params={"api_key": ROBOFLOW_API_KEY},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        )
        result = response.json()

        if "predictions" not in result:
            print(f"Roboflow response: {result}")
            return classify_local_fallback(image_path)

        class_map = {
            "alive": "alive", "dead": "dead", "no_plant": "no_plant",
            "hidup": "alive", "mati": "dead", "kosong": "no_plant"
        }

        predictions = result["predictions"]
        best = max(predictions, key=lambda x: x["confidence"])
        classification = class_map.get(best["class"].lower(), best["class"].lower())

        all_probs = {}
        for pred in predictions:
            cls = class_map.get(pred["class"].lower(), pred["class"].lower())
            all_probs[cls] = round(pred["confidence"], 3)

        return {
            "classification": classification,
            "confidence": round(best["confidence"], 3),
            "all_probabilities": all_probs,
            "source": "roboflow_api"
        }

    except Exception as e:
        print(f"Roboflow error: {e}")
        return classify_local_fallback(image_path)


def classify_local_fallback(image_path):
    try:
        import tensorflow as tf
        import numpy as np

        model = tf.keras.models.load_model('models/plant_classifier_final.keras')
        with open('models/class_names.json') as f:
            class_names = json.load(f)

        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.expand_dims(
            tf.keras.utils.img_to_array(img), 0) / 255.0
        preds = model.predict(img_array, verbose=0)[0]
        best_idx = int(tf.argmax(preds))

        return {
            "classification": class_names[best_idx],
            "confidence": round(float(preds[best_idx]), 3),
            "all_probabilities": {
                class_names[i]: round(float(preds[i]), 3)
                for i in range(len(class_names))
            },
            "source": "local_tensorflow"
        }
    except Exception as e:
        return {
            "classification": "no_plant",
            "confidence": 0.0,
            "all_probabilities": {},
            "source": "error",
            "error": str(e)
        }


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None

    if not path:
        for folder in [
            'dataset/validation/alive',
            'dataset/validation/dead',
            'dataset/validation/no_plant'
        ]:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder)
                         if f.lower().endswith('.jpg')]
                if files:
                    path = os.path.join(folder, files[0])
                    break

    if path:
        print(f"\nTest foto: {path}")
        result = classify_image_roboflow(path)
        labels = {
            'alive': 'TANAMAN HIDUP',
            'dead': 'TANAMAN MATI',
            'no_plant': 'TIDAK ADA TANAMAN'
        }
        print(f"Hasil     : {labels.get(result['classification'], result['classification'])}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Sumber    : {result['source']}")
    else:
        print("Tidak ada foto untuk ditest!")