from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import traceback

# إعداد اللغة لتجنب مشاكل الترميز
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model_pipeline = PredictionPipeline(self.filename)

    def predict(self, image_data):
        try:
            print("[DEBUG] Starting prediction process...")
            # حفظ الصورة المرسلة
            decodeImage(image_data, self.filename)
            print("[DEBUG] Image decoded and saved as:", self.filename)

            # تنفيذ التنبؤ
            result = self.model_pipeline.predict()
            print("[DEBUG] Prediction result:", result)

            return result
        except Exception as e:
            print("[ERROR in predict()]", e)
            traceback.print_exc()
            return {"error": str(e)}

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        print("[DEBUG] /predict endpoint called")

        if not request.json or 'image' not in request.json:
            print("[ERROR] No image found in request")
            return jsonify({"error": "No image data received"}), 400

        image_data = request.json['image']
        print("[DEBUG] Image data received (length):", len(image_data))

        result = clApp.predict(image_data)
        print("[DEBUG] /predict completed successfully")

        return jsonify(result)
    except Exception as e:
        print("[EXCEPTION in /predict]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[INFO] Starting Flask server...")
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)
