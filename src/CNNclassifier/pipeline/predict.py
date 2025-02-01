import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load and preprocess image
        test_image = load_img(self.filename, target_size=(300, 300))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # Class mapping
        classes = {0: "10", 1: "100", 2: "20", 3: "200", 4: "50", 5: "500", 6: "2000"}

        prediction = classes.get(result[0], "Unknown")

        return [{"image": prediction}]
