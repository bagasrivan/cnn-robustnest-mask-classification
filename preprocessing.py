import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

DATASET_PATH = "Dataset"
IMG_SIZE = (224, 224)

SEVERITY_LEVELS = {
    'Brighten': [1.1, 1.3, 1.5, 1.7, 1.9],
    'Darken': [0.9, 0.75, 0.6, 0.45, 0.3],
    'Blur': [3, 5, 7, 9, 11],
    'Low Compression': [90, 70, 50, 30, 10],
    'Rotate': [5, 10, 15, 20, 25]
}

# =========================
# SEED 
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =========================
# DEGRADATION FUNCTION
# =========================
def create_degraded_image(image, degradation_type="Normal", level=None):

    img = np.array(image)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if degradation_type == "Brighten":
        factor = level if level else 1.5
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == "Darken":
        factor = level if level else 0.6
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == "Blur":
        k = int(level if level else 7)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif degradation_type == "Low Compression":
        q = int(level if level else 50)
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img = cv2.imdecode(enc, 1)

    elif degradation_type == "Rotate":
        angle = level if level else 15
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32)

# =========================
# DATA LOADER
# =========================
def load_data_generators(batch_size=32):

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

    print("Train:", train_gen.samples)
    print("Val  :", val_gen.samples)

    # =========================
    # TEST SET
    # =========================
    test_eval = ImageDataGenerator(rescale=1./255)

    test_gen = test_eval.flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    test_generators = {"Normal": test_gen}

    def make_fn(deg, lvl):
        return lambda img: create_degraded_image(img, deg, lvl)

    for deg, levels in SEVERITY_LEVELS.items():
        for i, lvl in enumerate(levels, 1):

            key = f"{deg}_L{i}"

            gen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=make_fn(deg, lvl)
            ).flow_from_directory(
                os.path.join(DATASET_PATH, "test"),
                target_size=IMG_SIZE,
                batch_size=1,
                class_mode="categorical",
                shuffle=False
            )

            test_generators[key] = gen

    print("===== DATA LOADING COMPLETE =====")

    return train_gen, val_gen, test_generators