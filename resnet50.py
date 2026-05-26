import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocessing import load_data_generators


# ======================================================
# SEED SETUP (REPRODUCIBLE)
# ======================================================

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================================================
# MODEL TRAINING
# ======================================================

def create_model_and_train_finetuning(train_gen, val_gen, model_name):

    print(f"\n--- Building Model {model_name} ---")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # =========================
    # STAGE 1: TRAIN HEAD
    # =========================
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # =========================
    # LOGGING
    # =========================
    log_dir = f"logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)

    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    checkpoint = ModelCheckpoint(
        filepath=f'saved_models/best_{model_name}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3
    )

    print("\n--- Stage 1: Training Head ---")

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=[early_stopping, reduce_lr, tensorboard]
    )

    # =========================
    # STAGE 2: FINE TUNING
    # =========================

    print("\n--- Stage 2: Fine-Tuning ---")

    base_model.trainable = True

    for layer in base_model.layers[:140]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )

    # =========================
    # SAVE HISTORY
    # =========================
    os.makedirs("logs", exist_ok=True)
    np.save(f"logs/{model_name}_history.npy", history2.history)

    return model, history2


# ======================================================
# EVALUATION
# ======================================================

def evaluate_multiple_tests(model, test_generators, model_name):

    print(f"\n--- Evaluating {model_name} ---")

    model.load_weights(f'saved_models/best_{model_name}.keras')

    results = {}

    important_conditions = [
        'Normal',
        'Brighten_L1', 'Brighten_L3', 'Brighten_L5',
        'Darken_L1', 'Darken_L3', 'Darken_L5',
        'Blur_L1', 'Blur_L3', 'Blur_L5',
        'Low Compression_L1', 'Low Compression_L3', 'Low Compression_L5',
        'Rotate_L1', 'Rotate_L3', 'Rotate_L5'
    ]

    for name, gen in test_generators.items():

        print(f"\n--- Testing: {name} ---")

        gen.reset()

        preds = model.predict(gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        gen.reset()

        loss, acc = model.evaluate(gen, verbose=0)

        f1 = f1_score(gen.classes, y_pred, average='macro')

        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")

        print(classification_report(gen.classes, y_pred))

        if name in important_conditions:

            cm = confusion_matrix(gen.classes, y_pred)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            os.makedirs("confusion_matrices", exist_ok=True)

            plt.savefig(f"confusion_matrices/cm_{name}_{model_name}.png")
            plt.close()

        results[name] = {
            "accuracy": acc,
            "macro_f1": f1
        }

    return results


# ======================================================
# MAIN EXPERIMENT
# ======================================================

if __name__ == "__main__":

    seeds = [42, 123, 999]
    all_results = []

    for seed in seeds:

        print("\n==============================")
        print(f"SEED {seed}")
        print("==============================")

        set_seed(seed)

        train_gen, val_gen, test_eval, _ = load_data_generators()

        model, history = create_model_and_train_finetuning(
            train_gen,
            val_gen,
            model_name=f"resnet50_seed_{seed}"
        )

        # =========================
        # PLOT TRAINING
        # =========================

        os.makedirs("plots", exist_ok=True)

        plt.figure()
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.legend()
        plt.title(f"Accuracy Seed {seed}")
        plt.savefig(f"plots/acc_seed_{seed}.png")
        plt.close()

        # =========================
        # EVALUATE
        # =========================

        results = evaluate_multiple_tests(
            model,
            test_eval,
            f"resnet50_seed_{seed}"
        )

        all_results.append(results)

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print("\n===== FINAL RESULTS =====")

    conditions = all_results[0].keys()
    final_rows = []

    for c in conditions:

        accs = [r[c]["accuracy"] for r in all_results]
        f1s = [r[c]["macro_f1"] for r in all_results]

        print(f"\n{c}")
        print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        final_rows.append({
            "Condition": c,
            "Accuracy Mean": np.mean(accs),
            "Accuracy Std": np.std(accs),
            "F1 Mean": np.mean(f1s),
            "F1 Std": np.std(f1s)
        })

    os.makedirs("results", exist_ok=True)

    pd.DataFrame(final_rows).to_csv(
        "results/final_results.csv",
        index=False
    )

    print("\nSaved: results/final_results.csv")