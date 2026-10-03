# Handwritten Digit Recognition Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-green)

A machine-learning project for recognizing handwritten digits (0–9) using the **MNIST dataset** and comparing three classifiers:

- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM) with RBF kernel
- K-Nearest Neighbors (KNN)

This repository is the cleaned, executable version of the implementation documented in the accompanying project report.

## Project Objective

The objective is to automate handwritten digit recognition using supervised machine-learning techniques and compare different classification approaches in terms of accuracy and computational characteristics.

## Dataset

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 classes (digits 0–9)
- 28 × 28 grayscale images

For the model comparison, this implementation follows the project report and uses:

- **10,000 training samples**
- **2,000 test samples**

## Methodology

```text
MNIST
  ↓
28 × 28 grayscale image
  ↓
Flatten → 784 features
  ↓
Normalize pixels: [0,255] → [0,1]
  ↓
10,000 training / 2,000 test subset
  ↓
StandardScaler
  ↓
 ┌─────────────┬──────────────┬─────────────┐
 │     MLP     │     SVM      │     KNN     │
 └─────────────┴──────────────┴─────────────┘
                ↓
        Accuracy / Reports /
        Confusion Matrices
```

### Preprocessing

Each 28 × 28 image is flattened into a **784-dimensional vector**.

Pixel values are normalized:

```python
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0
```

The selected data is then standardized with `StandardScaler`.

```python
scaler = StandardScaler()

X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)
```

The scaler is fitted on the training data and then applied to the test data.

## Models

### 1. Multi-Layer Perceptron

```python
MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    max_iter=50,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

The three hidden layers allow the network to learn nonlinear relationships between pixel features.

### 2. Support Vector Machine

```python
SVC(
    kernel="rbf",
    C=5,
    gamma="scale",
    random_state=42
)
```

The RBF kernel allows the SVM to model nonlinear decision boundaries.

### 3. K-Nearest Neighbors

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="distance",
    n_jobs=-1
)
```

KNN predicts a digit using the nearest training examples.

## Results Reported in the Project

The original project report records the following results on the 2,000-sample held-out test subset:

| Model | Accuracy | Reported Training Time |
|---|---:|---:|
| MLP Neural Network | **93.25%** | 6.23 s |
| SVM (RBF Kernel) | **93.10%** | 12.44 s |
| KNN (k=5) | **89.15%** | 0.01 s |

### Important

These are the **results reported in the project report**. Running the repository performs the experiment again, so the exact accuracy and training time can vary with the Python/library versions, CPU and runtime environment.

The code does **not** hard-code these results.

## Project Structure

```text
handwritten-digit-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── project_metadata.json
│
├── src/
│   └── train.py
│
├── notebooks/
│   └── handwritten_digit_recognition.ipynb
│
├── results/
│   └── README.md
│
├── docs/
│   └── Handwritten_Digit_Recognition_Report.pdf
│
├── data/
│   └── .gitkeep
│
└── models/
    └── .gitkeep
```

## Installation

### 1. Clone

```bash
git clone https://github.com/YOUR-USERNAME/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```

The MNIST dataset will be downloaded through TensorFlow/Keras on the first run.

Generated files are placed in `results/`.

## Outputs

The program generates:

- `sample_images.png`
- `class_distribution.png`
- `model_accuracy_comparison.png`
- `mlp_training_curve.png`
- confusion matrices
- `model_results.csv`

## Interview Explanation

### 30-second explanation

> "My project is a handwritten digit recognition system using the MNIST dataset. I preprocess the 28×28 grayscale images by flattening them into 784 features, normalize the pixel values and standardize the data. I then compare three supervised learning algorithms: an MLP neural network, an RBF-kernel SVM and KNN. I evaluate them using accuracy, classification reports and confusion matrices."

### Why did you use three models?

> "I wanted to compare different approaches to classification. MLP can learn nonlinear patterns, SVM with an RBF kernel provides a strong nonlinear classifier, and KNN gives a simple distance-based baseline."

### Why flatten the image?

> "The models used in this experiment expect tabular feature vectors, so each 28×28 image is converted into a 784-dimensional vector."

### Why divide by 255?

> "The original grayscale pixels range from 0 to 255. Dividing by 255 converts them to the range 0 to 1, which makes the numerical input easier for machine-learning algorithms to process."

### Why StandardScaler?

> "After normalization, StandardScaler standardizes each feature using the training data. This is useful for algorithms such as MLP and SVM that are sensitive to feature scale."

### Why only 10,000 training samples?

> "The complete MNIST training set contains 60,000 images. The project used a 10,000-sample subset for the SVM and KNN comparison to keep computation manageable while still providing a representative dataset."

### Why did MLP perform well?

> "The MLP contains multiple hidden layers and can learn nonlinear relationships between the 784 pixel features."

### Why did KNN perform lower?

> "KNN relies on distance calculations. In a 784-dimensional feature space, distance-based methods can become less discriminative because of the curse of dimensionality."

### What would you improve?

> "I would use a CNN because handwritten digits are images and CNNs can preserve and learn spatial relationships between neighboring pixels. I would also add data augmentation and deploy the model through a Flask or Streamlit application."

## Important Interview Honesty

The accompanying college report contains screenshots/snippets of the implementation rather than the original source repository. Therefore this GitHub repository is a **clean, executable implementation faithful to the methodology and model configurations documented in the report**. It should not be described as a byte-for-byte copy of an original source file unless that original source is available.


## Interactive Handwriting Demo

The repository also includes a Streamlit demonstration where a user can draw a digit with the mouse and receive an MLP prediction.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train and save the MLP

The interactive application needs a trained model and scaler.

Run this once from the project root:

```bash
python src/train_model.py
```

This creates:

```text
models/
├── mlp_model.pkl
└── scaler.pkl
```

These files are generated locally and are intentionally ignored by Git because trained model binaries can be large and environment-dependent.

### Step 3 — Start the web application

```bash
streamlit run app.py
```

Streamlit will provide a local address, normally similar to:

```text
http://localhost:8501
```

Open it in your browser.

### Step 4 — Test it

1. Draw a digit from 0 to 9.
2. Click **Predict Digit**.
3. The application preprocesses the drawing.
4. The saved MLP predicts the digit.
5. The app displays the prediction and model confidence.

### Interactive Demo Pipeline

```text
Mouse Drawing
     ↓
Canvas Image
     ↓
Grayscale + Invert
     ↓
Detect Handwriting Bounding Box
     ↓
Crop + Preserve Aspect Ratio
     ↓
Center / Pad into 28 × 28
     ↓
Normalize / 255
     ↓
Flatten → 784 Features
     ↓
Saved StandardScaler
     ↓
Saved MLP
     ↓
Predicted Digit
```

### Important

The drawing application is an **interactive deployment/demo layer** added to the project. The original project report lists Flask/Streamlit deployment as future scope; the report's benchmark experiment itself evaluates MLP, SVM and KNN on the MNIST test subset.

### GitHub and Model Files

The `.gitignore` excludes:

```text
models/*.pkl
```

because the model files are generated artifacts. To reproduce them, run:

```bash
python src/train_model.py
```


## Future Scope

- CNN / LeNet-5
- Data augmentation
- Ensemble learning
- Flask/Streamlit deployment
- EMNIST or custom handwritten datasets

## Report

The full project report is available in:

```text
docs/Handwritten_Digit_Recognition_Report.pdf
```

## Author

**Debadrita Konar**  
B.Tech – Computer Science & Engineering
