from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from CNNclassifier.utils.common import decodeImage
from CNNclassifier.pipeline.predict import PredictionPipeline


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully!"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json.get("image")
        if not image:
            return jsonify({"error": "No image data received"}), 400

        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()

        return jsonify(result)  # ✅ Ensure JSON response is sent

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # ✅ Handle errors gracefully


if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host="0.0.0.0", port=8080)  # local host
    app.run(host="0.0.0.0", port=8080)  # for AWS
    # app.run(host="0.0.0.0", port=80)  # for AZURE
