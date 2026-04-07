import streamlit as st
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide")

st.title("Image Super Resolution using Bicubic Interpolation")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload a Low Resolution Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    low_res = cv2.imdecode(file_bytes, 1)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Slider (Upscaling Factor)
    # -----------------------------
    scale_factor = st.slider("Select Upscaling Factor", 2, 4, 2)

    # -----------------------------
    # Upscaling
    # -----------------------------
    height, width = low_res.shape[:2]
    new_width = width * scale_factor
    new_height = height * scale_factor

    bicubic = cv2.resize(low_res, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # -----------------------------
    # Sharpening (Novelty)
    # -----------------------------
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    sharpened = cv2.filter2D(bicubic, -1, kernel)

    # -----------------------------
    # Resize input for fair comparison
    # -----------------------------
    input_resized = cv2.resize(low_res, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # -----------------------------
    # Show Resolution Info
    # -----------------------------
    st.write(f"Output Resolution: {new_width} x {new_height}")

    # -----------------------------
    # BEFORE / AFTER SLIDER 🔥
    # -----------------------------
    st.subheader("Before vs After (Drag Slider)")

    image_comparison(
        img1=input_resized,
        img2=sharpened,
        label1="Low Resolution (Upscaled)",
        label2="Enhanced Output",
    )

    # -----------------------------
    # Side-by-side comparison
    # -----------------------------
    st.subheader("Detailed Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(input_resized, caption="Input (Upscaled for comparison)", width=300)

    with col2:
        st.image(bicubic, caption=f"Bicubic ({scale_factor}x)", width=300)

    with col3:
        st.image(sharpened, caption=f"Sharpened ({scale_factor}x)", width=300)

    # -----------------------------
    # Success Message
    # -----------------------------
    st.success("Upscaling Complete!")