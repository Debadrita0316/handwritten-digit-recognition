"""
Train and save the MLP model used by the interactive Streamlit demo.

The model follows the project report's preprocessing:
    MNIST -> flatten 28x28 to 784 -> normalize /255 -> StandardScaler -> MLP

Run:
    python src/train_model.py
"""

from pathlib import Path
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_SUBSET = 10000
RANDOM_STATE = 42


def main():
    print("Loading MNIST...")
    (x_train_raw, y_train), _ = mnist.load_data()

    # Flatten 28x28 images into 784 features.
    x_train = x_train_raw[:TRAIN_SUBSET].reshape(TRAIN_SUBSET, -1).astype("float32")

    # Normalize pixel values to [0, 1].
    x_train = x_train / 255.0
    y_train = y_train[:TRAIN_SUBSET]

    # Standardize using training data only.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    print("Training MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    model.fit(x_train_scaled, y_train)

    with open(MODELS_DIR / "mlp_model.pkl", "wb") as file:
        pickle.dump(model, file)

    with open(MODELS_DIR / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    print("\nSaved:")
    print(MODELS_DIR / "mlp_model.pkl")
    print(MODELS_DIR / "scaler.pkl")


if __name__ == "__main__":
    main()
