import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        print("[DEBUG] Initializing PredictionPipeline...")
        model_path = os.path.join("model", "model.h5")
        print(f"[DEBUG] Loading model from: {model_path}")
        self.model = load_model(model_path, compile=False)
        print("[DEBUG] Model loaded successfully!")

    def predict(self):
        print(f"[DEBUG] Starting prediction for file: {self.filename}")

        if not os.path.exists(self.filename):
            print(f"[ERROR] File not found: {self.filename}")
            return [{"error": "Image file not found"}]

        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        print("[DEBUG] Image preprocessed, running prediction...")
        result = np.argmax(self.model.predict(test_image), axis=1)
        print(f"[DEBUG] Raw model output: {result}")

        prediction = 'Tumor' if result[0] == 1 else 'Normal'
        print(f"[DEBUG] Final prediction: {prediction}")
        return [{"image": prediction}]
