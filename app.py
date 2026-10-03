"""
Interactive Handwritten Digit Recognition Demo

Run:
    python src/train_model.py
    streamlit run app.py

The drawing is converted into an MNIST-like 28x28 image before prediction:
    drawing -> grayscale -> invert -> crop -> preserve aspect ratio
    -> center/pad -> normalize -> flatten -> StandardScaler -> MLP
"""

from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "mlp_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ Handwritten Digit Recognition")
st.write(
    "Draw a digit from **0 to 9** in the box and click **Predict Digit**."
)


if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.warning(
        "Trained model files were not found. Run this once from the "
        "project root:\n\n"
        "`python src/train_model.py`"
    )
    st.stop()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_model()

st.subheader("Draw a digit")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    return_image_data=True,
)

predict_button = st.button(
    "🔍 Predict Digit",
    type="primary",
    use_container_width=True,
)


def preprocess_drawing(image_data):
    """
    Convert the 280x280 canvas drawing into an MNIST-like 28x28 image.

    MNIST digits occupy a relatively small centered region inside a
    28x28 image. Therefore, instead of resizing the entire canvas directly,
    we detect the drawn pixels, crop the digit, preserve its aspect ratio,
    pad it into a square, and center it.
    """

    # RGBA -> grayscale.
    gray = Image.fromarray(image_data.astype(np.uint8)).convert("L")

    # Canvas: black drawing on white background.
    # MNIST: bright digit on dark background.
    digit = 255 - np.array(gray, dtype=np.uint8)

    # Ignore very faint anti-aliased pixels/noise.
    mask = digit > 30

    if not np.any(mask):
        return None

    # Find bounding box of the actual handwritten digit.
    rows, cols = np.where(mask)

    top = rows.min()
    bottom = rows.max() + 1
    left = cols.min()
    right = cols.max() + 1

    cropped = digit[top:bottom, left:right]

    # Convert to PIL and preserve aspect ratio.
    cropped_image = Image.fromarray(cropped).convert("L")

    crop_width, crop_height = cropped_image.size

    # MNIST digits are generally contained within a smaller central area.
    # Fit the drawing into a 20x20 box while preserving proportions.
    target_size = 20

    scale = min(
        target_size / crop_width,
        target_size / crop_height,
    )

    new_width = max(1, int(round(crop_width * scale)))
    new_height = max(1, int(round(crop_height * scale)))

    resized = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    # Create a 28x28 black canvas.
    mnist_like = Image.new(
        "L",
        (28, 28),
        0,
    )

    # Put the digit in the center.
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    mnist_like.paste(
        resized,
        (x_offset, y_offset),
    )

    # Convert to [0,1], matching MNIST preprocessing.
    processed = np.array(mnist_like).astype("float32") / 255.0

    return processed


if predict_button:

    if canvas_result.image_data is None:
        st.error("Please draw a digit first.")
        st.stop()

    processed = preprocess_drawing(canvas_result.image_data)

    if processed is None:
        st.error("No digit was detected. Please draw a digit.")
        st.stop()

    # Flatten 28x28 -> 784 features.
    features = processed.reshape(1, -1)

    # Apply exactly the scaler used during MLP training.
    features_scaled = scaler.transform(features)

    # Predict.
    prediction = int(model.predict(features_scaled)[0])

    # Probability is useful as a confidence indicator, but it should not
    # be interpreted as a guaranteed probability of correctness.
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities)) * 100

    st.success(f"### Prediction: **{prediction}**")

    if confidence is not None:
        st.metric(
            "Model confidence",
            f"{confidence:.2f}%",
        )

    st.subheader("Processed MNIST-style 28×28 image")

    # Enlarge the 28x28 image only for visualization.
    preview = Image.fromarray(
        np.uint8(processed * 255)
    ).resize(
        (224, 224),
        Image.Resampling.NEAREST,
    )

    st.image(
        preview,
        width=224,
    )

    st.caption(
        "Preprocessing: crop handwritten pixels → preserve aspect ratio → "
        "center in 28×28 → normalize → flatten → StandardScaler → MLP."
    )

st.divider()

st.caption(
    "This interactive demo is an additional deployment layer for the "
    "MNIST handwritten-digit recognition project."
)
