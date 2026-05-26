import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from preprocessing import load_data_generators


# ======================================================
# SET RANDOM SEED
# ======================================================

def set_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================================================
# BUILD MODEL + TRAIN
# ======================================================

def create_model_and_train_finetuning(
        train_gen,
        val_gen,
        model_name='resnet50_robustness'
):

    print(f"\n--- Building Model {model_name} ---")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # ==================================================
    # STAGE 1 - TRAIN HEAD
    # ==================================================

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(3, activation='softmax')(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Stage 1: Training Head ---")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=[

            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )

        ]
    )

    # ==================================================
    # STAGE 2 - FINE TUNING
    # ==================================================

    print("\n--- Stage 2: Fine-Tuning ---")

    base_model.trainable = True

    fine_tune_at = 140

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs('saved_models', exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=f'saved_models/best_{model_name}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ]
    )

    return model, history


# ======================================================
# EVALUATION
# ======================================================

def evaluate_multiple_tests(
        model,
        test_generators,
        model_name
):

    print(f"\n--- Evaluating Model {model_name} ---")

    model.load_weights(
        f'saved_models/best_{model_name}.keras'
    )

    print("✅ Best weights loaded.")

    results = {}

    important_conditions = [
        # BASELINE
        'Normal',

        # BRIGHTEN
        'Brighten_L1',
        'Brighten_L3',
        'Brighten_L5',

        # DARKEN
        'Darken_L1',
        'Darken_L3',
        'Darken_L5',

        # BLUR
        'Blur_L1',
        'Blur_L3',
        'Blur_L5',

        # LOW COMPRESSION
        'Low Compression_L1',
        'Low Compression_L3',
        'Low Compression_L5',

        # ROTATE
        'Rotate_L1',
        'Rotate_L3',
        'Rotate_L5'
    ]

    for deg_type, test_gen in test_generators.items():

        print(f"\n--- Testing on: {deg_type} ---")

        test_gen.reset()

        Y_pred = model.predict(
            test_gen,
            verbose=0
        )

        y_pred = np.argmax(Y_pred, axis=1)

        test_gen.reset()

        test_loss, test_acc = model.evaluate(
            test_gen,
            verbose=0
        )

        macro_f1 = f1_score(
            test_gen.classes,
            y_pred,
            average='macro'
        )

        print(f"Accuracy  : {test_acc:.4f}")
        print(f"Macro-F1  : {macro_f1:.4f}")

        class_names = list(
            test_gen.class_indices.keys()
        )

        print("\nClassification Report:")

        print(
            classification_report(
                test_gen.classes,
                y_pred,
                target_names=class_names
            )
        )

        # ==================================================
        # CONFUSION MATRIX (ONLY IMPORTANT CONDITIONS)
        # ==================================================

        if deg_type in important_conditions:

            cm = confusion_matrix(
                test_gen.classes,
                y_pred
            )

            plt.figure(figsize=(7, 6))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )

            plt.title(
                f'Confusion Matrix - {deg_type} ResNet50'
            )

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            plt.tight_layout()

            os.makedirs(
                'confusion_matrices',
                exist_ok=True
            )

            save_path = (
                f'confusion_matrices/'
                f'cm_{deg_type}_{model_name}.png'
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight'
            )

            print(f"📁 Saved: {save_path}")

            plt.close()

        # ==================================================
        # SAVE RESULTS
        # ==================================================

        results[deg_type] = {
            'accuracy': test_acc,
            'macro_f1': macro_f1
        }

    return results


# ======================================================
# MAIN
# ======================================================

if __name__ == '__main__':

    seeds = [42, 123, 999]

    all_results = []

    for seed in seeds:

        print("\n====================================")
        print(f"RUNNING EXPERIMENT - SEED {seed}")
        print("====================================")

        # ==================================================
        # SET SEED
        # ==================================================

        set_seed(seed)

        # ==================================================
        # LOAD DATA
        # ==================================================

        train_gen, val_gen, test_gens_eval, _ = \
            load_data_generators()

        # ==================================================
        # TRAIN MODEL
        # ==================================================

        model, history = create_model_and_train_finetuning(
            train_gen,
            val_gen,
            model_name=f'resnet50_seed_{seed}'
        )

        # ==================================================
        # SAVE TRAINING CURVE
        # ==================================================

        os.makedirs('plots', exist_ok=True)

        plt.figure(figsize=(8, 5))

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.legend([
            'Train Accuracy',
            'Validation Accuracy'
        ])

        plt.title(
            f'Training Curve - Seed {seed}'
        )

        plt.tight_layout()

        plt.savefig(
            f'plots/training_curve_seed_{seed}.png',
            dpi=300
        )

        plt.close()

        # ==================================================
        # EVALUATE
        # ==================================================

        results = evaluate_multiple_tests(
            model,
            test_gens_eval,
            f'resnet50_seed_{seed}'
        )

        all_results.append(results)

    # ======================================================
    # FINAL MEAN ± STD
    # ======================================================

    print("\n====================================")
    print("FINAL RESULTS (MEAN ± STD)")
    print("====================================")

    degradation_types = all_results[0].keys()

    final_rows = []

    for deg in degradation_types:

        acc_scores = [
            run[deg]['accuracy']
            for run in all_results
        ]

        f1_scores = [
            run[deg]['macro_f1']
            for run in all_results
        ]

        acc_mean = np.mean(acc_scores)
        acc_std = np.std(acc_scores)

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)

        print(f"\n{deg}")

        print(
            f"Accuracy : "
            f"{acc_mean:.4f} ± {acc_std:.4f}"
        )

        print(
            f"Macro-F1 : "
            f"{f1_mean:.4f} ± {f1_std:.4f}"
        )

        final_rows.append({

            'Condition': deg,

            'Accuracy Mean': acc_mean,
            'Accuracy Std': acc_std,

            'MacroF1 Mean': f1_mean,
            'MacroF1 Std': f1_std

        })

    # ======================================================
    # SAVE CSV
    # ======================================================

    os.makedirs('results', exist_ok=True)

    df = pd.DataFrame(final_rows)

    csv_path = (
        'results/resnet50_final_results.csv'
    )

    df.to_csv(csv_path, index=False)

    print(f"\n📁 CSV saved to: {csv_path}")