"""
Handwritten Digit Recognition Using Machine Learning

This implementation follows the methodology and model configurations
documented in the accompanying project report.

Models:
    1. Multi-Layer Perceptron (MLP)
    2. Support Vector Machine (SVM - RBF)
    3. K-Nearest Neighbors (KNN)

Dataset:
    MNIST - 28x28 grayscale handwritten digits

Important:
    The report used a 10,000-sample training subset and a 2,000-sample
    test subset for the SVM/KNN comparison. This script follows that setup.
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras.datasets import mnist

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_SUBSET = 10000
TEST_SUBSET = 2000
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Step 1 - Load MNIST Dataset
# ---------------------------------------------------------------------

def load_dataset():
    """Load the MNIST dataset using TensorFlow/Keras."""

    (x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()

    print("Training data shape:", x_train_raw.shape)
    print("Test data shape:", x_test_raw.shape)
    print("Unique classes:", np.unique(y_train))

    return x_train_raw, y_train, x_test_raw, y_test


# ---------------------------------------------------------------------
# Step 2 - Visualize Sample Images
# ---------------------------------------------------------------------

def visualize_sample_images(x_train, y_train):
    """Display and save a small set of MNIST samples."""

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x_train[i], cmap="gray")
        ax.set_title(f"Digit {y_train[i]}")
        ax.axis("off")

    fig.suptitle(
        "Sample MNIST Images (Digits 0-9)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "sample_images.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()


# ---------------------------------------------------------------------
# Step 3 - Explore Class Distribution
# ---------------------------------------------------------------------

def plot_class_distribution(y_train, y_test):
    """Plot training and test class distributions."""

    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(
        x=list(range(10)),
        y=train_counts,
        ax=axes[0],
    )
    axes[0].set_title("Training Class Distribution")
    axes[0].set_xlabel("Digit Class")
    axes[0].set_ylabel("Count")

    sns.barplot(
        x=list(range(10)),
        y=test_counts,
        ax=axes[1],
    )
    axes[1].set_title("Test Class Distribution")
    axes[1].set_xlabel("Digit Class")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "class_distribution.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()


# ---------------------------------------------------------------------
# Step 4 - Preprocessing
# ---------------------------------------------------------------------

def preprocess_data(x_train_raw, y_train, x_test_raw, y_test):
    """
    Flatten 28x28 images to 784-dimensional vectors,
    normalize pixel values to [0, 1], select the project subset,
    and standardize features.
    """

    # Flatten 28x28 images to 784-dimensional vectors.
    x_train_flat = x_train_raw.reshape(
        x_train_raw.shape[0], -1
    ).astype("float32")

    x_test_flat = x_test_raw.reshape(
        x_test_raw.shape[0], -1
    ).astype("float32")

    # Normalize pixel values from [0, 255] to [0, 1].
    x_train_norm = x_train_flat / 255.0
    x_test_norm = x_test_flat / 255.0

    # Use the subset described in the project report.
    x_tr = x_train_norm[:TRAIN_SUBSET]
    y_tr = y_train[:TRAIN_SUBSET]

    x_te = x_test_norm[:TEST_SUBSET]
    y_te = y_test[:TEST_SUBSET]

    # Standardization: zero mean and unit variance.
    scaler = StandardScaler()

    x_tr_scaled = scaler.fit_transform(x_tr)
    x_te_scaled = scaler.transform(x_te)

    print("\nPreprocessing complete.")
    print("Training subset:", x_tr_scaled.shape)
    print("Test subset:", x_te_scaled.shape)
    print("Feature vector dimension:", x_tr_scaled.shape[1])

    return x_tr_scaled, y_tr, x_te_scaled, y_te


# ---------------------------------------------------------------------
# Step 5 - Train MLP
# ---------------------------------------------------------------------

def train_mlp(x_train, y_train, x_test, y_test):
    """Train and evaluate the Multi-Layer Perceptron."""

    print("\nTraining MLP Neural Network...")
    start = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    mlp.fit(x_train, y_train)

    training_time = time.time() - start
    prediction = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)

    print(f"MLP Training Time : {training_time:.2f}s")
    print(f"MLP Accuracy      : {accuracy * 100:.2f}%")

    return mlp, prediction, accuracy, training_time


# ---------------------------------------------------------------------
# Step 6 - Train SVM
# ---------------------------------------------------------------------

def train_svm(x_train, y_train, x_test, y_test):
    """Train and evaluate the RBF-kernel Support Vector Machine."""

    print("\nTraining SVM Classifier...")
    start = time.time()

    svm = SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        random_state=RANDOM_STATE,
    )

    svm.fit(x_train, y_train)

    training_time = time.time() - start
    prediction = svm.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)

    print(f"SVM Training Time : {training_time:.2f}s")
    print(f"SVM Accuracy      : {accuracy * 100:.2f}%")

    return svm, prediction, accuracy, training_time


# ---------------------------------------------------------------------
# Step 7 - Train KNN
# ---------------------------------------------------------------------

def train_knn(x_train, y_train, x_test, y_test):
    """Train and evaluate K-Nearest Neighbors."""

    print("\nTraining KNN Classifier...")
    start = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        weights="distance",
        n_jobs=-1,
    )

    knn.fit(x_train, y_train)

    training_time = time.time() - start
    prediction = knn.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)

    print(f"KNN Training Time : {training_time:.2f}s")
    print(f"KNN Accuracy      : {accuracy * 100:.2f}%")

    return knn, prediction, accuracy, training_time


# ---------------------------------------------------------------------
# Step 8 - Accuracy Comparison
# ---------------------------------------------------------------------

def plot_accuracy_comparison(results):
    """Create the model accuracy comparison chart."""

    models = results["Model"].tolist()
    accuracies = results["Accuracy (%)"].tolist()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        models,
        accuracies,
        edgecolor="black",
        linewidth=0.8,
    )

    plt.ylabel("Accuracy (%)")
    plt.title(
        "Model Accuracy Comparison on MNIST",
        fontsize=14,
        fontweight="bold",
    )

    plt.ylim(min(85, min(accuracies) - 3), 100)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{accuracy:.2f}%",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "model_accuracy_comparison.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()


# ---------------------------------------------------------------------
# Step 9 - MLP Training Curve
# ---------------------------------------------------------------------

def plot_mlp_training_curve(mlp):
    """
    Plot the training loss curve.

    sklearn's MLPClassifier exposes loss_curve_ and validation_scores_.
    validation_scores_ is validation accuracy, so it is shown separately
    rather than being incorrectly labelled as validation loss.
    """

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        mlp.loss_curve_,
        label="Training Loss",
    )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Training Loss")
    ax1.set_title(
        "MLP Neural Network - Training Curve",
        fontsize=14,
        fontweight="bold",
    )

    if hasattr(mlp, "validation_scores_"):
        ax2 = ax1.twinx()
        ax2.plot(
            mlp.validation_scores_,
            linestyle="--",
            label="Validation Accuracy",
        )
        ax2.set_ylabel("Validation Accuracy")

    fig.tight_layout()
    plt.savefig(
        RESULTS_DIR / "mlp_training_curve.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()


# ---------------------------------------------------------------------
# Step 10 - Confusion Matrices
# ---------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create a confusion matrix for a model."""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("HANDWRITTEN DIGIT RECOGNITION USING MACHINE LEARNING")
    print("=" * 70)

    # Load data.
    x_train_raw, y_train, x_test_raw, y_test = load_dataset()

    # Visualizations.
    visualize_sample_images(x_train_raw, y_train)
    plot_class_distribution(y_train, y_test)

    # Preprocessing.
    x_train, y_train_subset, x_test, y_test_subset = preprocess_data(
        x_train_raw,
        y_train,
        x_test_raw,
        y_test,
    )

    # Train all three models.
    mlp, mlp_pred, mlp_acc, mlp_time = train_mlp(
        x_train,
        y_train_subset,
        x_test,
        y_test_subset,
    )

    svm, svm_pred, svm_acc, svm_time = train_svm(
        x_train,
        y_train_subset,
        x_test,
        y_test_subset,
    )

    knn, knn_pred, knn_acc, knn_time = train_knn(
        x_train,
        y_train_subset,
        x_test,
        y_test_subset,
    )

    # Final comparison table.
    results = pd.DataFrame(
        {
            "Model": [
                "MLP Neural Network",
                "SVM (RBF Kernel)",
                "KNN (k=5)",
            ],
            "Accuracy (%)": [
                mlp_acc * 100,
                svm_acc * 100,
                knn_acc * 100,
            ],
            "Training Time (s)": [
                mlp_time,
                svm_time,
                knn_time,
            ],
        }
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    results.to_csv(
        RESULTS_DIR / "model_results.csv",
        index=False,
    )

    # Visualizations.
    plot_accuracy_comparison(results)
    plot_mlp_training_curve(mlp)

    plot_confusion_matrix(
        y_test_subset,
        mlp_pred,
        "MLP Neural Network",
    )

    plot_confusion_matrix(
        y_test_subset,
        svm_pred,
        "SVM RBF Kernel",
    )

    plot_confusion_matrix(
        y_test_subset,
        knn_pred,
        "KNN k5",
    )

    # Classification reports.
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORTS")
    print("=" * 70)

    print("\n--- MLP Neural Network ---")
    print(classification_report(
        y_test_subset,
        mlp_pred,
        digits=4,
    ))

    print("\n--- SVM (RBF Kernel) ---")
    print(classification_report(
        y_test_subset,
        svm_pred,
        digits=4,
    ))

    print("\n--- KNN (k=5) ---")
    print(classification_report(
        y_test_subset,
        knn_pred,
        digits=4,
    ))

    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
