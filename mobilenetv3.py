import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from preprocessing import load_data_generators, set_seed


# ======================================================
# TRAIN + MODEL
# ======================================================
def build_and_train(train_gen, val_gen, name):

    base = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(3, activation='softmax')(x)

    model = Model(base.input, out)

    model.compile(
        optimizer=Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # =========================
    # TRAINING HEAD
    # =========================
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )

    print("\n--- Stage 1 ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=[early_stop, reduce_lr]
    )

    # =========================
    # FINE TUNING
    # =========================
    base.trainable = True

    for l in base.layers[:60]:
        l.trainable = False

    model.compile(
        optimizer=Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f"best_{name}.h5",
        monitor='val_accuracy',
        save_best_only=True
    )

    print("\n--- Stage 2 ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[ckpt, early_stop, reduce_lr]
    )

    return model, f"best_{name}.h5"


# ======================================================
# EVALUATION
# ======================================================
def evaluate(model, test_generators, model_name):

    print(f"\n--- EVALUATING {model_name} ---")

    model.load_weights(f"best_{model_name}.h5")

    results = {}

    important = [
        'Normal',
        'Brighten_L1','Brighten_L3','Brighten_L5',
        'Darken_L1','Darken_L3','Darken_L5',
        'Blur_L1','Blur_L3','Blur_L5',
        'Low Compression_L1','Low Compression_L3','Low Compression_L5',
        'Rotate_L1','Rotate_L3','Rotate_L5'
    ]

    for name, gen in test_generators.items():

        gen.reset()

        preds = model.predict(gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        acc = model.evaluate(gen, verbose=0)[1]
        f1 = f1_score(gen.classes, y_pred, average='macro')

        print(f"{name} | Acc: {acc:.4f} | F1: {f1:.4f}")

        # =========================
        # CONFUSION MATRIX
        # =========================
        if name in important:

            cm = confusion_matrix(gen.classes, y_pred)

            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d')

            plt.title(f"{name} - MobileNetV3")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            os.makedirs("cm", exist_ok=True)
            plt.savefig(f"cm/cm_{name}_{model_name}.png")
            plt.close()

        print(classification_report(gen.classes, y_pred))

        results[name] = {
            "accuracy": acc,
            "f1": f1
        }

    return results


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    all_results = []

    for seed in [42, 123, 999]:

        print("\n====================")
        print("SEED", seed)
        print("====================")

        set_seed(seed)

        train, val, test = load_data_generators(batch_size=32)

        model, path = build_and_train(train, val, f"mobilenet_{seed}")

        results = evaluate(model, test, f"mobilenet_{seed}")

        all_results.append(results)


    # ======================================================
    # FINAL RESULT
    # ======================================================
    print("\n===== FINAL RESULT =====")

    keys = all_results[0].keys()

    for k in keys:

        accs = [r[k]["accuracy"] for r in all_results]
        f1s = [r[k]["f1"] for r in all_results]

        print(f"\n{k}")
        print(f"Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")