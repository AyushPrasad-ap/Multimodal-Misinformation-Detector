from __future__ import annotations

import tempfile
from pathlib import Path
from textwrap import dedent

import streamlit as st
import torch
from PIL import Image

from src.inference import Predictor, build_model
from src.preprocessing import ImageProcessor, TextProcessor
from src.utils.config import load_config
from src.utils.device import get_device

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"

config = load_config(CONFIG_PATH)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Multimodal Misinformation Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        margin-top: 2rem;
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 100%
        );
        color: white;
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #d1d5db;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Section cards */
    .section-card {
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    /* Prediction cards */
    .prediction-real {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #86efac;
        background: #f0fdf4;
        text-align: center;
    }

    .prediction-fake {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #fca5a5;
        background: #fef2f2;
        text-align: center;
    }

    .prediction-label {
        color: #266197;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }

    .prediction-confidence {
        font-size: 1rem;
        color: #4b5563;
    }

    /* Metric cards */
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        text-align: center;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model loading
# ============================================================


@st.cache_resource(show_spinner=False)
def load_predictor() -> tuple[Predictor, torch.device]:

    device = get_device(config["training"]["device"])

    text_config = config["models"]["text"]
    image_config = config["models"]["image"]
    fusion_config = config["models"]["fusion"]

    text_processor = TextProcessor(
        model_name=text_config["name"],
        max_length=text_config["max_length"],
    )

    image_processor = ImageProcessor()

    model = build_model(
        text_model_name=text_config["name"],
        text_freeze=text_config["freeze"],
        image_freeze=image_config["freeze"],
        hidden_dim=fusion_config["hidden_dim"],
        dropout=fusion_config["dropout"],
        num_classes=config["project"]["num_classes"],
        device=device,
    )

    checkpoint_path = PROJECT_ROOT / config["paths"]["checkpoint_dir"] / "best_model.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    labels = {
        int(class_id): label_name
        for class_id, label_name in config["project"]["labels"].items()
    }

    predictor = Predictor(
        model=model,
        text_processor=text_processor,
        image_processor=image_processor,
        device=device,
        labels=labels,
    )

    return predictor, device


# ============================================================
# Header
# ============================================================

st.markdown(
    dedent("""
<div class="hero">
<div class="hero-title">
🔎 Multimodal Misinformation Detector
</div>

<p class="hero-subtitle">
Analyze a news headline and its associated image using
a multimodal deep learning model combining DistilBERT
and ResNet-18 feature representations.
</p>
</div>
"""),
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("About the Model")

    st.markdown(
        """
        This application uses a multimodal neural network that
        jointly analyzes textual and visual information.
        """
    )

    st.divider()

    st.subheader("Architecture")

    st.markdown(
        """
        **Text Encoder**

        DistilBERT

        **Image Encoder**

        ResNet-18

        **Fusion**

        Concatenated 1280-D embeddings

        **Classifier**

        Fully connected neural network
        """
    )

    st.divider()

    st.subheader("Evaluation")

    st.metric("Test Accuracy", "80.93%")
    st.metric("F1 Score", "76.63%")
    st.metric("ROC-AUC", "88.50%")

    st.divider()

    st.caption("Inference uses the best validation-loss checkpoint.")


# ============================================================
# Input section
# ============================================================

st.subheader("Analyze a Multimodal Sample")

col_text, col_image = st.columns(
    [1.25, 1],
    gap="large",
)


with col_text:
    st.markdown("### 📝 Text")

    text = st.text_area(
        "Enter a news headline or claim",
        placeholder=(
            "Example: Scientists discover a new method "
            "to improve solar panel efficiency..."
        ),
        height=180,
        label_visibility="collapsed",
    )

    st.caption(
        "Provide the headline, claim, or textual content associated with the image."
    )


with col_image:
    st.markdown("### 🖼️ Image")

    uploaded_file = st.file_uploader(
        "Upload an associated image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        preview = Image.open(uploaded_file)

        st.image(
            preview,
            caption="Uploaded image",
            use_container_width=True,
        )


# ============================================================
# Analyze
# ============================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Content",
    type="primary",
    use_container_width=True,
)


if analyze:
    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not text or not text.strip():
        st.warning("Please enter a news headline or claim.")
        st.stop()

    if uploaded_file is None:
        st.warning("Please upload an image associated with the text.")
        st.stop()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    with st.spinner("Loading multimodal model and analyzing input..."):
        try:
            predictor, device = load_predictor()

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())

                temp_image_path = Path(temp_file.name)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = predictor.predict(
                text=text.strip(),
                image_path=temp_image_path,
            )

        except Exception as exc:  # noqa: BLE001
            st.error("An error occurred while processing the input.")

            st.exception(exc)

            st.stop()

        finally:
            if "temp_image_path" in locals():
                temp_image_path.unlink(missing_ok=True)

    # ========================================================
    # Results
    # ========================================================

    prediction = result["prediction"].upper()
    confidence = result["confidence"]

    probability_real = result["probability_real"]
    probability_fake = result["probability_fake"]

    st.subheader("Analysis Result")

    if prediction == "REAL":
        st.markdown(
            dedent(f"""
<div class="prediction-real">
<div class="prediction-confidence">MODEL PREDICTION</div>
<div class="prediction-label">
✓ REAL
</div>
<div class="prediction-confidence">
Confidence: {confidence * 100:.2f}%
</div>
</div>
"""),
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            dedent(f"""
<div class="prediction-fake">
<div class="prediction-confidence">MODEL PREDICTION</div>
<div class="prediction-label">
⚠ FAKE
</div>
<div class="prediction-confidence">
Confidence: {confidence * 100:.2f}%
</div>
</div>
"""),
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # Probability breakdown
    # --------------------------------------------------------

    col_real, col_fake = st.columns(2)

    with col_real:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Probability — REAL
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(probability_real)

        st.markdown(f"**{probability_real * 100:.2f}%**")

    with col_fake:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Probability — FAKE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(probability_fake)

        st.markdown(f"**{probability_fake * 100:.2f}%**")

    # --------------------------------------------------------
    # Technical information
    # --------------------------------------------------------

    with st.expander("Technical Details"):
        st.write(
            {
                "Prediction": result["prediction"],
                "Class ID": result["class_id"],
                "Confidence": f"{confidence:.4f}",
                "Real Probability": f"{probability_real:.4f}",
                "Fake Probability": f"{probability_fake:.4f}",
                "Inference Device": str(device),
                "Text Encoder": "DistilBERT",
                "Image Encoder": "ResNet-18",
            }
        )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">

        Multimodal Misinformation Detector ·
        PyTorch · Hugging Face · Torchvision · Streamlit

    </div>
    """,
    unsafe_allow_html=True,
)
