# preprocessing.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import random

DATASET_PATH = "Dataset"
IMG_SIZE = (224, 224)

SEVERITY_LEVELS = {
    'Brighten': [1.1, 1.3, 1.5, 1.7, 1.9],
    'Darken': [0.9, 0.75, 0.6, 0.45, 0.3],
    'Blur': [3, 5, 7, 9, 11],
    'Low Compression': [90, 70, 50, 30, 10],
    'Rotate': [5, 10, 15, 20, 25]
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_degraded_image(image, degradation_type="Normal", level=None):

    img = np.array(image)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if degradation_type == "Brighten":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * level, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == "Darken":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * level, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == "Blur":
        k = int(level)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif degradation_type == "Low Compression":
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(level)])
        img = cv2.imdecode(enc, 1)

    elif degradation_type == "Rotate":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), level, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def load_data_generators(batch_size=64):

    print("\n===== LOADING DATA =====")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    test_eval = ImageDataGenerator(rescale=1./255)

    test_generators_eval = {}

    test_generators_eval["Normal"] = test_eval.flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    test_generators_visual = {}

    def make_fn(deg, lvl):
        return lambda img: create_degraded_image(img, deg, lvl)

    for deg, levels in SEVERITY_LEVELS.items():
        for i, lvl in enumerate(levels, 1):

            key = f"{deg}_L{i}"

            gen_eval = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=make_fn(deg, lvl)
            ).flow_from_directory(
                os.path.join(DATASET_PATH, "test"),
                target_size=IMG_SIZE,
                batch_size=1,
                class_mode="categorical",
                shuffle=False
            )

            gen_vis = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=make_fn(deg, lvl)
            ).flow_from_directory(
                os.path.join(DATASET_PATH, "test"),
                target_size=IMG_SIZE,
                batch_size=9,
                class_mode="categorical",
                shuffle=True
            )

            test_generators_eval[key] = gen_eval
            test_generators_visual[key] = gen_vis

    print("===== DATA LOADING COMPLETE =====")

    return train_gen, val_gen, test_generators_eval, test_generators_visual