# preprocessing.py
import os
import numpy as np
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


# =========================
# DEGRADATION FUNCTION (SAFE)
# =========================
def create_degraded_image(image, deg="Normal", lvl=None):

    img = np.array(image)

    if deg == "Normal":
        return img

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if deg == "Brighten":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * lvl, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif deg == "Darken":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * lvl, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif deg == "Blur":
        k = int(lvl)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif deg == "Low Compression":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(lvl)]
        _, enc = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(enc, 1)

    elif deg == "Rotate":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), lvl, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


# =========================
# DATA LOADER (STABLE VERSION)
# =========================
def load_data_generators(batch_size=64):

    print("\n===== LOADING DATA =====")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    # =========================
    # TEST NORMAL
    # =========================
    test_eval = {}

    test_eval["Normal"] = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    # =========================
    # TEST DEGRADED
    # =========================
    for deg, levels in SEVERITY_LEVELS.items():
        for i, lvl in enumerate(levels, 1):

            key = f"{deg}_L{i}"

            test_eval[key] = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda img, d=deg, l=lvl: create_degraded_image(img, d, l)
            ).flow_from_directory(
                os.path.join(DATASET_PATH, "test"),
                target_size=IMG_SIZE,
                batch_size=1,
                class_mode="categorical",
                shuffle=False
            )

    print("===== DATA LOADING COMPLETE =====")

    return train_gen, val_gen, test_eval