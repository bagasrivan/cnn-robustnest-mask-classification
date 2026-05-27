import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import load_data_generators, create_degraded_image, DATASET_PATH

def set_system_seed(seed):
    """Mengunci seluruh generator angka acak sistem demi validitas eksperimen"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_mobilenetv3_model():
    """Membangun arsitektur dasar model MobileNetV3Large"""
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions), base_model

def train_model_per_seed(train_gen, val_gen, model_save_name):
    model, base_model = build_mobilenetv3_model()

    # ====== Tahap 1: Transfer Learning ======
    print("-> Memulai Tahap 1: Training Head Layer...")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_gen, validation_data=val_gen, epochs=25, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
    )

    # ====== Tahap 2: Fine-Tuning ======
    print("-> Memulai Tahap 2: Fine-Tuning Sebagian Layer...")
    base_model.trainable = True
    fine_tune_at = 60
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(model_save_name, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    model.fit(train_gen, validation_data=val_gen, epochs=50, verbose=1,
              callbacks=[checkpoint, early_stopping, reduce_lr])

    model.load_weights(model_save_name)
    return model

def evaluate_robustness_levels(model, seed_name):
    """Mengevaluasi model ke dalam 5 tingkatan degradasi lingkungan secara otomatis"""
    degradation_types = ['Brighten', 'Darken', 'Blur', 'Low Compression', 'Rotate']
    seed_results = {}

    # 1. Evaluasi Data Uji Kondisi Normal
    test_datagen_normal = ImageDataGenerator(rescale=1./255)
    test_gen_normal = test_datagen_normal.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'), target_size=(224, 224),
        batch_size=1, class_mode='categorical', shuffle=False, verbose=0
    )
    _, normal_acc = model.evaluate(test_gen_normal, verbose=0)
    seed_results['Normal_Level_0'] = normal_acc

    # 2. Evaluasi Kondisi Modifikasi Degradasi (Level 1-5)
    for deg_type in degradation_types:
        for sev in range(1, 6):
            test_preprocessing_function = lambda img, d=deg_type, s=sev: create_degraded_image(img, degradation_type=d, severity=s)
            
            test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=test_preprocessing_function)
            test_gen = test_datagen.flow_from_directory(
                os.path.join(DATASET_PATH, 'test'), target_size=(224, 224),
                batch_size=1, class_mode='categorical', shuffle=False, verbose=0
            )
            
            _, test_acc = model.evaluate(test_gen, verbose=0)
            key_name = f"{deg_type}_Level_{sev}"
            seed_results[key_name] = test_acc
            print(f"[{seed_name}] {deg_type} Level {sev} -> Akurasi: {test_acc:.4f}")

    return seed_results


if __name__ == '__main__':
    SEEDS = [42, 123, 999]
    all_runs_summary = {}

    for current_seed in SEEDS:
        print(f"\n========================================================")
        print(f"🏃 RUNNING MOBILENETV3 EXPERIMENT - SEED {current_seed}")
        print(f"========================================================")
        
        set_system_seed(current_seed)
        train_gen, val_gen = load_data_generators(current_seed=current_seed)
        
        model_name = f'best_mobilenetv3_seed_{current_seed}.keras'
        model = train_model_per_seed(train_gen, val_gen, model_name)
        
        seed_label = f"Seed_{current_seed}"
        all_runs_summary[seed_label] = evaluate_robustness_levels(model, seed_label)

    # === RINGKASAN DATA AKHIR REPEATED RUN ===
    print("\n==========================================================")
    print("📊 MOBILENETV3 REPEATED RUNS SUMMARY REPORT")
    print("==========================================================")
    print(f"{'Kondisi Eksperimen':<25} | {'Seed 42':<10} | {'Seed 123':<10} | {'Seed 999':<10} | {'Rata-rata':<10}")
    print("-" * 75)
    
    all_keys = all_runs_summary['Seed_42'].keys()
    for key in all_keys:
        acc_42 = all_runs_summary['Seed_42'][key]
        acc_123 = all_runs_summary['Seed_123'][key]
        acc_999 = all_runs_summary['Seed_999'][key]
        mean_acc = np.mean([acc_42, acc_123, acc_999])
        
        print(f"{key:<25} | {acc_42:.4f}     | {acc_123:.4f}      | {acc_999:.4f}      | {mean_acc:.4f}")