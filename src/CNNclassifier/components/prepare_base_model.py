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
        Constructs a sequential model and saves it to the specified path in the configuration.
        """
        # Define preprocessing and augmentation layers
        resize_and_rescale = tf.keras.layers.Rescaling(
            1.0 / 255
        )  # Normalize pixel values
        data_augmentation = tf.keras.Sequential(
            [  # Example data augmentation
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
            ]
        )

        # Build the sequential model
        self.model = tf.keras.Sequential(
            [
                resize_and_rescale,
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
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(self.config.params_classes, activation="softmax"),
            ]
        )

        # Build the model by passing a dummy input shape
        self.model.build(input_shape=(None,) + self.config.params_image_size)

        # Display the model summary
        self.model.summary()

        # Save the model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, learning_rate=None):
        """
        Adds custom layers to the base model for classification.
        """
        # Add custom layers for fine-tuning
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(
            flatten_in
        )

        # Create the full model
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        # Compile the full model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        # Display the model summary
        full_model.summary()
        return full_model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
