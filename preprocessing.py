import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "Dataset"
IMG_SIZE = (224, 224)
SEED = 42

SEVERITY_LEVELS = {
    'Brighten': [1.1, 1.3, 1.5, 1.7, 1.9],
    'Darken': [0.9, 0.75, 0.6, 0.45, 0.3],
    'Blur': [3, 5, 7, 9, 11],
    'Low Compression': [90, 70, 50, 30, 10],
    'Rotate': [5, 10, 15, 20, 25]
}

# =========================================================
# SEED
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# NORMALIZATION (IMPORTANT)
# =========================================================
def normalize(img):
    return img.astype(np.float32) / 255.0


# =========================================================
# IMAGE DEGRADATION (OPTIMIZED)
# =========================================================

def create_degraded_image(image, degradation_type="Normal", level=None):

    img = np.array(image)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if degradation_type == "Brighten":
        factor = float(level or 1.5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == "Darken":
        factor = float(level or 0.6)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == "Blur":
        k = int(level or 7)
        k = k if k % 2 == 1 else k + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif degradation_type == "Low Compression":
        q = int(level or 50)
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img = cv2.imdecode(enc, 1)

    elif degradation_type == "Rotate":
        angle = float(level or 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return normalize(img)


# =========================================================
# DATA LOADER (FAST VERSION)
# =========================================================

def load_data_generators(batch_size=64):

    print("\n===== FAST DATA LOADING =====")

    # =========================
    # TRAIN (WITH AUGMENTATION)
    # =========================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # =========================
    # VAL (NO AUGMENTATION)
    # =========================
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    print(f"Train samples: {train_gen.samples}")
    print(f"Val samples  : {val_gen.samples}")

    # =========================
    # TEST BASE (NORMAL ONLY FIRST)
    # =========================
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    test_generators_eval = {"Normal": test_gen}
    test_generators_visual = {}

    # =========================
    # DEGRADATION GENERATORS
    # =========================

    def make_fn(deg, lvl):
        def fn(img):
            return create_degraded_image(img, deg, lvl)
        return fn

    for deg, levels in SEVERITY_LEVELS.items():
        for i, lvl in enumerate(levels, 1):

            key = f"{deg}_L{i}"
            print(f"Loading {key}")

            fn = make_fn(deg, lvl)

            gen_eval = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=fn
            ).flow_from_directory(
                os.path.join(DATASET_PATH, "test"),
                target_size=IMG_SIZE,
                batch_size=1,
                class_mode="categorical",
                shuffle=False
            )

            test_generators_eval[key] = gen_eval

            gen_vis = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=fn
            ).flow_from_directory(
                os.path.join(DATASET_PATH, "test"),
                target_size=IMG_SIZE,
                batch_size=9,
                class_mode="categorical",
                shuffle=True
            )

            test_generators_visual[key] = gen_vis

    print("\n===== DATA LOADING COMPLETE =====")

    return train_gen, val_gen, test_generators_eval, test_generators_visual


# =========================================================
# VISUALIZATION
# =========================================================

def visualize_sample_data(generator, title="Sample"):

    images, labels = next(generator)
    class_names = list(generator.class_indices.keys())

    plt.figure(figsize=(8, 8))
    plt.suptitle(title)

    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")

    plt.tight_layout()
    plt.show()