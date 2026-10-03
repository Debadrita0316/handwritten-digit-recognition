# Handwritten Digit Recognition Using Machine Learning

A machine-learning project for recognizing handwritten digits (0–9) using the **MNIST dataset** and comparing three classifiers:

- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM) with RBF kernel
- K-Nearest Neighbors (KNN)

## Project Objective

The objective of this project is to recognize handwritten digits using supervised machine-learning techniques and compare different classification algorithms based on their performance.

## Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 classes representing digits 0–9
- 28 × 28 grayscale images

For the model comparison, the project uses:

- **10,000 training samples**
- **2,000 test samples**

## Methodology

The overall workflow of the project is:

```text
MNIST Dataset
      ↓
28 × 28 Grayscale Images
      ↓
Flatten Images
      ↓
784 Features
      ↓
Normalize Pixel Values
      ↓
Select Training/Test Subset
      ↓
StandardScaler
      ↓
 ┌──────────────┬──────────────┬──────────────┐
 │     MLP      │     SVM      │     KNN      │
 └──────────────┴──────────────┴──────────────┘
                ↓
        Model Evaluation
                ↓
 Accuracy / Classification Report /
      Confusion Matrix
```

## Data Preprocessing

The original MNIST images have dimensions of **28 × 28 pixels**.

### 1. Flattening

Each image is converted from a 28 × 28 matrix into a **784-dimensional feature vector**.

```python
X_train_flat = X_train_raw.reshape(
    X_train_raw.shape[0], -1
).astype("float32")

X_test_flat = X_test_raw.reshape(
    X_test_raw.shape[0], -1
).astype("float32")
```

### 2. Pixel Normalization

The original pixel values range from **0 to 255**.

They are normalized to the range **0 to 1** by dividing by 255.

```python
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0
```

### 3. Training and Testing Subset

The project uses:

- 10,000 training samples
- 2,000 test samples

```python
X_train_subset = X_train_norm[:10000]
y_train_subset = y_train[:10000]

X_test_subset = X_test_norm[:2000]
y_test_subset = y_test[:2000]
```

### 4. Feature Standardization

`StandardScaler` is used to standardize the features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_subset)
X_test_scaled = scaler.transform(X_test_subset)
```

The scaler is fitted only on the training data and then applied to the test data.

This prevents information from the test set from being used during training.

## Machine Learning Models

The project compares three supervised machine-learning algorithms.

### 1. Multi-Layer Perceptron

The MLP is a feed-forward neural network with three hidden layers.

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

#### MLP Architecture

```text
784 Input Features
       ↓
256 Neurons
       ↓
128 Neurons
       ↓
64 Neurons
       ↓
10 Output Classes
```

The ReLU activation function is used in the hidden layers and Adam is used as the optimization algorithm.

### 2. Support Vector Machine

The project uses an SVM with an RBF kernel.

```python
SVC(
    kernel="rbf",
    C=5,
    gamma="scale",
    random_state=42
)
```

The RBF kernel allows the SVM to learn nonlinear decision boundaries.

### 3. K-Nearest Neighbors

The project uses KNN with:

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="distance",
    n_jobs=-1
)
```

The algorithm classifies a new image based on its nearest training examples.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix
- Training Time

### Accuracy

Accuracy measures the percentage of correctly classified test images.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

for each digit class.

### Confusion Matrix

A confusion matrix shows how many samples from each actual digit class were predicted as each class.

It helps identify which digits are commonly confused with each other.

## Results

The project report records the following results on the 2,000-sample test subset:

| Model | Accuracy | Reported Training Time |
|---|---:|---:|
| MLP Neural Network | **93.25%** | 6.23 s |
| SVM (RBF Kernel) | **93.10%** | 12.44 s |
| KNN (k=5) | **89.15%** | 0.01 s |

Training time can vary depending on the hardware and software environment.

## Interactive Handwriting Recognition

The project also contains a **Streamlit-based interactive application**.

The application allows the user to draw a handwritten digit using the mouse and receive a prediction from the trained MLP model.

### Interactive Demo Workflow

```text
User Draws Digit
       ↓
Canvas Image
       ↓
Grayscale Conversion
       ↓
Color Inversion
       ↓
Detect Handwritten Region
       ↓
Crop Digit
       ↓
Preserve Aspect Ratio
       ↓
Center Digit
       ↓
Resize / Pad to 28 × 28
       ↓
Normalize Pixel Values
       ↓
Flatten to 784 Features
       ↓
StandardScaler
       ↓
Trained MLP Model
       ↓
Predicted Digit
```

### Train the Model for the Demo

From the project root, run:

```bash
python src/train_model.py
```

This creates:

```text
models/
├── mlp_model.pkl
└── scaler.pkl
```

### Start the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser.

The default local address is:

```text
http://localhost:8501
```

### Using the Application

1. Draw a digit from 0 to 9.
2. Click **Predict Digit**.
3. The application preprocesses the drawing.
4. The processed image is converted into the required 28 × 28 format.
5. The trained MLP predicts the digit.
6. The predicted digit and model confidence are displayed.

## Project Structure

```text
handwritten-digit-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── project_metadata.json
├── app.py
│
├── src/
│   ├── train.py
│   └── train_model.py
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

### Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

## Run the Machine Learning Experiment

Run:

```bash
python src/train.py
```

The MNIST dataset will be downloaded automatically through TensorFlow/Keras during the first run.

The program trains and evaluates:

- MLP
- SVM
- KNN

Generated results are saved inside the `results/` directory.

## Generated Outputs

The project generates:

```text
results/
│
├── sample_images.png
├── class_distribution.png
├── model_accuracy_comparison.png
├── mlp_training_curve.png
├── mlp_neural_network_confusion_matrix.png
├── svm_rbf_kernel_confusion_matrix.png
├── knn_k5_confusion_matrix.png
└── model_results.csv
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Pillow
- Jupyter Notebook

## Applications

Handwritten digit recognition can be used in areas such as:

- Handwritten form digitization
- Postal and document processing
- Cheque processing
- Educational applications
- Digit recognition systems
- Accessibility applications

## Future Scope

The project can be further improved by:

- Using Convolutional Neural Networks (CNNs)
- Applying data augmentation
- Using deeper neural network architectures
- Training on larger datasets
- Using EMNIST or custom handwritten datasets
- Improving the interactive application
- Deploying the application online

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*.

2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.

3. Abadi, M. et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning*.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.

5. MNIST Dataset.

6. Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*.

## Author

**Debadrita Konar**

B.Tech – Computer Science & Engineering
