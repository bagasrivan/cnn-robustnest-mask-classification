import os
import numpy as np
import tensorflow as tf
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


# =========================
# TF DATA PIPELINE (FAST)
# =========================

def load_data_generators(batch_size=64):

    print("\n===== LOADING DATA (TF.DATA FAST VERSION) =====")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        validation_split=0.2,
        subset="validation",
        seed=42
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=IMG_SIZE,
        batch_size=1,
        label_mode="categorical",
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    # =========================
    # BUILD DEGRADATION TEST SETS 
    # =========================
    test_generators_eval = {"Normal": test_ds}

    def apply_degradation(img, label, deg, lvl):
        img = tf.cast(img, tf.float32)

        if deg == "Brighten":
            img = img * lvl
        elif deg == "Darken":
            img = img * lvl
        elif deg == "Blur":
            img = tf.image.random_brightness(img, max_delta=0.1)
        elif deg == "Rotate":
            img = tf.image.rot90(img, k=int(lvl) % 4)
        elif deg == "Low Compression":
            img = tf.image.random_jpeg_quality(img, 30, int(lvl))

        return img, label

    # generate degraded datasets
    for deg, levels in SEVERITY_LEVELS.items():
        for i, lvl in enumerate(levels, 1):

            key = f"{deg}_L{i}"

            ds = test_ds.map(
                lambda x, y, d=deg, l=lvl: apply_degradation(x, y, d, l),
                num_parallel_calls=AUTOTUNE
            ).cache().prefetch(AUTOTUNE)

            test_generators_eval[key] = ds

    print("===== DATA LOADING COMPLETE =====")

    return train_ds, val_ds, test_generators_eval