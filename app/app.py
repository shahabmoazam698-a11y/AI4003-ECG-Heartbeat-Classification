
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt


# =========================================================
# 1. AUTOMATIC PROJECT DIRECTORY DETECTION
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent

possible_directories = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
    CURRENT_DIR.parent.parent,
    CURRENT_DIR.parent.parent.parent
]

PROJECT_DIR = None

for directory in possible_directories:
    if (directory / "src" / "model.py").exists():
        PROJECT_DIR = directory
        break

if PROJECT_DIR is None:
    st.error("Cannot locate the project src/model.py file.")
    st.stop()

SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from model import ECGNet
from prepare import prepare_data


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ECG Heartbeat Classifier",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# 3. CLASS INFORMATION
# =========================================================

CLASS_NAMES = {
    0: "N - Normal",
    1: "S - Supraventricular Ectopic",
    2: "V - Ventricular Ectopic",
    3: "F - Fusion",
    4: "Q - Unknown / Paced"
}


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = RESULTS_DIR / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = ECGNet()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# =========================================================
# 5. LOAD TEST DATA
# =========================================================

@st.cache_resource
def load_test_data():

    data = prepare_data()

    return data["X_test"], data["y_test"]


# =========================================================
# 6. PREDICTION FUNCTION
# =========================================================

def predict_heartbeat(model, signal):

    signal = signal.reshape(1, 187).float()

    with torch.no_grad():

        logits = model(signal)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return predicted_class, probabilities[0].numpy()


# =========================================================
# 7. APPLICATION HEADER
# =========================================================

st.title("❤️ ECG Heartbeat Classification System")

st.markdown("""
### Artificial Neural Network Based ECG Classification

This application classifies ECG heartbeat signals into five
categories using a custom neural network trained with PyTorch.

**Network architecture:** 187 → 128 → 64 → 5
""")

st.divider()


# =========================================================
# 8. MODEL LOADING
# =========================================================

try:

    model = load_model()

    st.success("Trained model loaded successfully.")

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# 9. INPUT SELECTION
# =========================================================

st.subheader("ECG Signal Input")

input_option = st.radio(
    "Select input method:",
    [
        "Select a test sample",
        "Upload an ECG CSV file"
    ],
    horizontal=True
)

signal = None
actual_class = None


# =========================================================
# 10. SELECT TEST DATASET SAMPLE
# =========================================================

if input_option == "Select a test sample":

    try:

        X_test, y_test = load_test_data()

        sample_index = st.number_input(
            "Select test sample number",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1
        )

        sample_index = int(sample_index)

        signal = X_test[sample_index]

        actual_class = int(y_test[sample_index])

        st.info(
            f"Actual class: {CLASS_NAMES[actual_class]}"
        )

    except Exception as error:

        st.error(f"Test dataset loading failed: {error}")
        st.stop()


# =========================================================
# 11. UPLOAD ECG CSV
# =========================================================

else:

    uploaded_file = st.file_uploader(
        "Upload CSV containing one ECG heartbeat",
        type=["csv"],
        help="The CSV must have one row containing 187 numeric values."
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file,
                header=None
            )

            if uploaded_data.shape != (1, 187):

                st.error(
                    "Invalid CSV format. "
                    "Expected exactly one row and 187 columns."
                )

                st.stop()

            uploaded_data = uploaded_data.apply(
                pd.to_numeric,
                errors="coerce"
            )

            if uploaded_data.isna().any().any():

                st.error(
                    "CSV contains missing or non-numeric values."
                )

                st.stop()

            signal = torch.tensor(
                uploaded_data.iloc[0].to_numpy(
                    dtype="float32"
                ),
                dtype=torch.float32
            )

            st.success("ECG signal uploaded successfully.")

        except Exception as error:

            st.error(f"CSV upload error: {error}")
            st.stop()


# =========================================================
# 12. ECG WAVEFORM VISUALIZATION
# =========================================================

if signal is not None:

    st.divider()

    st.subheader("ECG Waveform")

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(
        signal.numpy(),
        linewidth=1.5
    )

    ax.set_title("ECG Heartbeat Signal")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)


    # =====================================================
    # 13. CLASSIFICATION BUTTON
    # =====================================================

    if st.button(
        "Classify Heartbeat",
        type="primary",
        use_container_width=True
    ):

        predicted_class, probabilities = predict_heartbeat(
            model,
            signal
        )

        st.divider()

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Heartbeat: "
            f"{CLASS_NAMES[predicted_class]}"
        )

        if actual_class is not None:

            if predicted_class == actual_class:

                st.success("The model predicted correctly.")

            else:

                st.warning(
                    f"Actual class: {CLASS_NAMES[actual_class]}"
                )


        # =================================================
        # 14. PROBABILITY VISUALIZATION
        # =================================================

        st.subheader("Class Prediction Probabilities")

        probability_data = pd.DataFrame({
            "Class": [
                CLASS_NAMES[i] for i in range(5)
            ],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_data.set_index("Class")
        )

        st.subheader("Detailed Probabilities")

        for index, probability in enumerate(probabilities):

            st.write(
                f"**{CLASS_NAMES[index]}:** "
                f"{probability * 100:.2f}%"
            )


# =========================================================
# 15. PROJECT INFORMATION
# =========================================================

with st.expander("About this project"):

    st.write("""
    **Assignment:** AI4003 Applied Machine Learning

    **Dataset:** MIT-BIH Heartbeat Dataset

    **Input:** 187 ECG signal measurements

    **Hidden layers:** 128 and 64 neurons

    **Output:** Five heartbeat classes

    **Framework:** PyTorch

    **Interface:** Streamlit
    """)

st.divider()

st.caption(
    "AI4003 Applied Machine Learning | "
    "ECG Heartbeat Classification Project"
)
