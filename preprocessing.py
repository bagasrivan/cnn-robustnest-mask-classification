import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Tentukan path dataset Anda
DATASET_PATH = 'Dataset'

# Kamus untuk menentukan parameter spesifik tiap level (1-5)
# Level 3 adalah nilai asli/default dari kode lama Anda
SEVERITY_CONFIG = {
    'Brighten':         {1: 1.1,  2: 1.3,  3: 1.5,  4: 1.7,  5: 1.9},
    'Darken':           {1: 0.9,  2: 0.7,  3: 0.6,  4: 0.4,  5: 0.2},
    'Blur':             {1: 3,    2: 5,    3: 7,    4: 9,    5: 11}, 
    'Low Compression':  {1: 80,   2: 65,   3: 50,   4: 35,   5: 20}, 
    'Rotate':           {1: 5,    2: 10,   3: 15,   4: 20,   5: 25}
}

def create_degraded_image(image, degradation_type='Normal', severity=3):
    """
    Membuat degradasi gambar berdasarkan jenis dan tingkat keparahan (severity 1-5).
    """
    image = np.array(image)
    degraded_img = image.copy()

    if degradation_type == 'Normal':
        return degraded_img.astype(np.float32)

    # Ambil nilai parameter berdasarkan jenis degradasi dan level keparahannya
    level_val = SEVERITY_CONFIG[degradation_type][severity]

    # Convert RGB → BGR untuk pemrosesan OpenCV
    if degraded_img.shape[-1] == 3:
        degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR)

    if degradation_type == 'Brighten':
        hsv = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * level_val, 0, 255)
        degraded_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == 'Darken':
        hsv = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * level_val, 0, 255)
        degraded_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif degradation_type == 'Blur':
        degraded_img = cv2.GaussianBlur(degraded_img, (level_val, level_val), 0)

    elif degradation_type == 'Low Compression':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), level_val]
        _, encimg = cv2.imencode('.jpg', degraded_img, encode_param)
        degraded_img = cv2.imdecode(encimg, 1)

    elif degradation_type == 'Rotate':
        h, w = degraded_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, level_val, 1.0)
        degraded_img = cv2.warpAffine(
            degraded_img, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    # Convert kembali BGR → RGB
    degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)
    return degraded_img.astype(np.float32)


def load_data_generators(batch_size=32, img_height=224, img_width=224, current_seed=123):
    """
    Memuat generator training & validation menggunakan seed yang ditentukan agar eksperimen konsisten.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=current_seed
    )
    
    # Generator untuk data validasi (normal, tanpa augmentasi)
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'validation'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=current_seed
    )

    return train_generator, validation_generator