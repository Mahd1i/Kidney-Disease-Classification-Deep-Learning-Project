import os
from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Load the updated base model and unfreeze last layers for fine-tuning"""
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Fine-tuning: افعل unfreeze لآخر 50 طبقة إذا ممكن
        for layer in self.model.layers[-50:]:
            layer.trainable = True

    def train_valid_generator(self):
        """Create training and validation generators with advanced augmentation"""
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.2
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation generator (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator (strong augmentation)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest',
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save trained model"""
        model.save(path)

    def train(self):
        """Train the model with advanced callbacks and fine-tuning"""
        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Optimizer with lower learning rate for fine-tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.trained_model_path,
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,  # الآن يأخذ عدد الـ epochs من ملف الإعدادات
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
