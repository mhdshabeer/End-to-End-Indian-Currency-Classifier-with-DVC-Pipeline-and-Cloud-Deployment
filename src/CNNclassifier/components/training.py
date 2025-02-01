from CNNclassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
from keras.callbacks import ReduceLROnPlateau


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.base_model_path)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20,  # 20% for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        # Validation Data Generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,  # Keep validation data order consistent
            **dataflow_kwargs,
        )

        # Training Data Generator with or without Augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,  # Shuffle training data for randomness
            **dataflow_kwargs,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        print(self.train_generator.class_indices)
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )
        lr_reduction = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        )
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=[callback_list, lr_reduction],
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
