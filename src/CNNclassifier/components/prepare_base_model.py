import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from CNNclassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
)


class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        """
        Constructs a sequential model, compiles it, and saves it to the specified path in the configuration.
        """

        # Build the sequential model
        self.model = tf.keras.Sequential(
            [
                # Sequential wrapper for Conv2D layers
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=self.config.params_image_size,
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(self.config.params_classes, activation="softmax"),
            ]
        )

        # Build the model by passing a dummy input shape
        self.model.build(input_shape=(None, 300, 300, 3))

        # Display the model summary
        self.model.summary()

        # Compile the model (as classification layers already exist in the model)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        # Save the compiled model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
