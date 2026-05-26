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
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from preprocessing import load_data_generators


# ======================================================
# SPEED OPTIMIZATION
# ======================================================
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy("float32")


# ======================================================
# SEED
# ======================================================
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================================================
# MODEL BUILD + TRAIN
# ======================================================
def create_model_and_train_finetuning(train_gen, val_gen, model_name):

    print(f"\n--- Building Model {model_name} ---")

    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # =========================
    # STAGE 1 (HEAD TRAINING FAST)
    # =========================
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
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
        histogram_freq=0,
        write_graph=False
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2
    )

    print("\n--- Stage 1 Training (FAST) ---")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25, 
        callbacks=[early_stopping, reduce_lr, tensorboard],
        verbose=1
    )

    # =========================
    # STAGE 2 (FINE TUNING LIGHT)
    # =========================
    print("\n--- Stage 2 Fine-tuning ---")

    base_model.trainable = True

    for layer in base_model.layers[:60]:  
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs("saved_models", exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=f'saved_models/best_{model_name}.h5',  
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,  
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
        verbose=1
    )

    np.save(f"logs/{model_name}_history.npy", history.history)

    return model, history


# ======================================================
# EVALUATION
# ======================================================
def evaluate_multiple_tests(model, test_generators, model_name):

    print(f"\n--- Evaluating {model_name} ---")

    model.load_weights(f'saved_models/best_{model_name}.h5')

    results = {}

    important_conditions = [
        'Normal',
        'Brighten_L1','Brighten_L3','Brighten_L5',
        'Darken_L1','Darken_L3','Darken_L5',
        'Blur_L1','Blur_L3','Blur_L5',
        'Low Compression_L1','Low Compression_L3','Low Compression_L5',
        'Rotate_L1','Rotate_L3','Rotate_L5'
    ]

    for name, gen in test_generators.items():

        print(f"\n--- Testing: {name} ---")

        gen.reset()

        preds = model.predict(gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        gen.reset()

        acc = model.evaluate(gen, verbose=0)[1]
        f1 = f1_score(gen.classes, y_pred, average='macro')

        print(f"Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f}")

        print(classification_report(gen.classes, y_pred))

        if name in important_conditions:

            cm = confusion_matrix(gen.classes, y_pred)

            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

            plt.title(f"{name} - MobileNetV3")
            plt.xlabel("Pred")
            plt.ylabel("True")

            os.makedirs("confusion_matrices", exist_ok=True)
            plt.savefig(f"confusion_matrices/cm_{name}_{model_name}.png")
            plt.close()

        results[name] = {
            "accuracy": acc,
            "macro_f1": f1
        }

    return results


# ======================================================
# MAIN
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
            model_name=f"mobilenetv3_seed_{seed}"
        )

        # =========================
        # PLOT
        # =========================
        os.makedirs("plots", exist_ok=True)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['train','val'])
        plt.title(f"MobileNetV3 Seed {seed}")
        plt.savefig(f"plots/mobilenetv3_seed_{seed}.png")
        plt.close()

        # =========================
        # EVALUATION
        # =========================
        results = evaluate_multiple_tests(
            model,
            test_eval,
            f"mobilenetv3_seed_{seed}"
        )

        all_results.append(results)

    # =========================
    # FINAL RESULTS
    # =========================
    print("\n===== FINAL RESULTS =====")

    conditions = all_results[0].keys()
    final_rows = []

    for c in conditions:

        accs = [r[c]["accuracy"] for r in all_results]
        f1s = [r[c]["macro_f1"] for r in all_results]

        print(f"\n{c}")
        print(f"Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        final_rows.append({
            "Condition": c,
            "Accuracy Mean": np.mean(accs),
            "Accuracy Std": np.std(accs),
            "F1 Mean": np.mean(f1s),
            "F1 Std": np.std(f1s)
        })

    os.makedirs("results", exist_ok=True)

    pd.DataFrame(final_rows).to_csv(
        "results/mobilenetv3_final_results.csv",
        index=False
    )

    print("\nSaved results.")