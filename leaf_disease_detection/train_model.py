import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE    = 224
NUM_CLASSES = 37
SEED        = 42


def build_model(num_classes: int):
    """ResNet50 backbone + custom classification head."""
    base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False      # freeze for phase-1

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs), base


def get_generators(dataset_path: str, batch_size: int):
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    valid_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        seed=SEED,
    )
    valid_gen = valid_aug.flow_from_directory(
        os.path.join(dataset_path, 'valid'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        seed=SEED,
    )
    return train_gen, valid_gen


def train(dataset_path: str, epochs: int, batch_size: int, output_path: str):
    print(f"\n{'='*60}")
    print(f"  LeafScan — Model Training")
    print(f"  Dataset  : {dataset_path}")
    print(f"  Epochs   : {epochs}   Batch size : {batch_size}")
    print(f"  Output   : {output_path}")
    print(f"{'='*60}\n")

    train_gen, valid_gen = get_generators(dataset_path, batch_size)
    model, base = build_model(NUM_CLASSES)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    callbacks = [
        ModelCheckpoint(output_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.3, patience=3, monitor='val_loss', verbose=1),
    ]

    phase1_epochs = min(5, epochs)

    # ── Phase 1 : train head, backbone frozen ────────────────────
    print(f"\n[Phase 1] Training classification head ({phase1_epochs} epochs, backbone frozen)…\n")
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=phase1_epochs,
        callbacks=callbacks,
    )

    if epochs > phase1_epochs:
        # ── Phase 2 : unfreeze top 30 backbone layers ─────────────
        print(f"\n[Phase 2] Fine-tuning top ResNet50 layers ({epochs - phase1_epochs} epochs)…\n")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=epochs,
            initial_epoch=phase1_epochs,
            callbacks=callbacks,
        )

    print(f"\n✅  Model saved to: {output_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LeafScan plant disease model')
    parser.add_argument('--dataset_path', default='./dataset',
                        help='Path to dataset folder with train/ and valid/ sub-directories')
    parser.add_argument('--epochs',     type=int, default=15, help='Total training epochs (default 15)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default 32)')
    parser.add_argument('--output',     default='detector/ml_model/plant_disease_model.h5',
                        help='Output model path')
    args = parser.parse_args()
    train(args.dataset_path, args.epochs, args.batch_size, args.output)