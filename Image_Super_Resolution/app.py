import streamlit as st
import cv2
import numpy as np
import io
import sys
import os

# Seamless integration: Append path to seamlessly import the backend logic
sys.path.append(os.path.dirname(__file__))
from src.main import (
    upscale_bicubic, 
    upscale_ai, 
    clarify_ai_image,
    apply_sharpening
)

from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide", page_title="AI Super Resolution Studio", page_icon="✨")

# -----------------------------
# Stylish Header
# -----------------------------
st.title("🌟 AI Super Resolution Studio")
st.markdown(
    "Easily upload images to neatly upscale them using completely integrated Python logic. "
    "Switch between rigid math (Bicubic) or our natively embedded neural networks (ESPCN)!"
)

# -----------------------------
# Neat Sidebar Configuration
# -----------------------------
st.sidebar.header("⚙ Processing Controls")
upscale_method = st.sidebar.radio(
    "Select Upscaling Engine:", 
    ["Deep Learning AI (ESPCN)", "Mathematical Bicubic"]
)

# Updated slider parameters natively requested
num_cols = st.sidebar.columns([1])
scale_factor = st.sidebar.slider(
    "Select Upscaling Multiplier (Zoom Factor):", 
    min_value=2, max_value=8, value=4, step=1,
    help="Higher numbers mean larger outputs, drastically increasing dimensions."
)

st.sidebar.markdown("---")
apply_sharpening_bool = st.sidebar.checkbox(
    "✨ Post-Process: Apply Output Clarification Filter", 
    value=True,
    help="This applies Unsharp Masking/Laplacian algorithms dynamically to unblur upscale outputs automatically."
)

st.sidebar.markdown("---")
apply_pixel_art = st.sidebar.checkbox(
    "👾 Enable Pixel-Art De-blockify Mode", 
    value=False,
    help="If your image is made of huge bloated pixel blocks, this destroys them cleanly before intelligent smoothing."
)

if apply_pixel_art:
    block_size = st.sidebar.number_input("Estimated Block Width (Pixels):", min_value=1, max_value=100, value=16)
else:
    block_size = 1

# -----------------------------
# Upload Image Area
# -----------------------------
st.subheader("1. 📸 Upload Low Resolution Image")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Safely digest uploaded bytes back into fully-fledged openCV representations
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    low_res = cv2.imdecode(file_bytes, 1)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
    
    height, width = low_res.shape[:2]
    target_size = (int(width * scale_factor), int(height * scale_factor))

    # Info banner letting user organically gauge exactly what sizes are outputted
    st.info(f"Target Output Dimension rendering out to: **{target_size[0]} x {target_size[1]}** pixels")

    if st.button("🚀 Upscale Image Now", use_container_width=True, type="primary"):
        with st.spinner(f"Super-Res engine crunching {target_size[0]}x{target_size[1]} frame sizes natively inline... Please wait."):
            
            # --- De-Blockify Processing ---
            if apply_pixel_art and block_size > 1:
                tiny_w = max(1, width // block_size)
                tiny_h = max(1, height // block_size)
                work_img = cv2.resize(low_res, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA)
                st.info(f"Analyzed structural resolution. Image block architecture gracefully minimized to: **{tiny_w}x{tiny_h} px**.")
            else:
                work_img = low_res

            # -----------------------------
            # Interlinked Core Algorithm Handling
            # -----------------------------
            if "Deep Learning" in upscale_method:
                # Notice we explicitly use work_img which handles both normal sizes and crushed sizes securely
                output = upscale_ai(work_img, target_size)
                if apply_sharpening_bool:
                    # Unsharp mask for explicitly smoothing AI logic
                    output = clarify_ai_image(output)
            else:
                output = upscale_bicubic(work_img, target_size)
                if apply_sharpening_bool:
                    # Intense standard filter
                    output = apply_sharpening(output)

            # Reorient identical scaling boundaries solely for side-by-side review in the slider
            input_resized_for_view = cv2.resize(low_res, target_size, interpolation=cv2.INTER_NEAREST)

            st.success("Successfully computed scaling matrix!")
            st.divider()

            # -----------------------------
            # SLIDER UI: Interactive Clean Review
            # -----------------------------
            st.subheader("2. 🔍 Interactive Result Inspection (Before vs After)")
            
            image_comparison(
                img1=input_resized_for_view,
                img2=output,
                label1=f"Original (Nearest {scale_factor}x)",
                label2=f"{upscale_method} {scale_factor}x Output",
                width=1000 if target_size[0] > 1000 else target_size[0],  # Bound constraints logically to keep it visually flawless
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

            st.divider()

            # -----------------------------
            # Local Download Extraction Button
            # -----------------------------
            st.subheader("3. 📥 Download Finalized Export")
            st.markdown("Your image has been fully upscaled. Acquire the original `.png` asset securely directly onto your machine.")
            
            # Format it completely cleanly sequentially binary buffered
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            io_buf = io.BytesIO(buffer)

            file_name_tag = "ESPCN_Super_Resolution" if "Deep Learning" in upscale_method else "Bicubic_Interpolation"
            file_name = f"{file_name_tag}_{target_size[0]}x{target_size[1]}.png"

            st.download_button(
                label=f"⬇ Download Processed PNG (Resolution: {target_size[0]}x{target_size[1]})",
                data=io_buf,
                file_name=file_name,
                mime="image/png",
                type="primary",
                use_container_width=True
            )
else:
    st.markdown("Waiting for your image...")
