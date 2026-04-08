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
    upscale_lanczos, 
    clarify_image,
    apply_sharpening,
    compute_metrics,
    upscale_nearest
)
import pandas as pd

from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide", page_title="AI Super Resolution Studio", page_icon="✨")

# -----------------------------
# Stylish Header
# -----------------------------
st.title("🌟 AI Super Resolution Studio")
st.markdown(
    "Easily upload images to neatly upscale them using completely integrated mathematically rigorous IVP logic. "
    "Switch between standard rigid math (Bicubic) or our advanced structural interpolations (Lanczos-4)!"
)

# -----------------------------
# Neat Sidebar Configuration
# -----------------------------
st.sidebar.header("🕹️ Primary Operation Mode")
operation_mode = st.sidebar.radio(
    "Select Workflow Type:",
    ["🚀 Practical Upscale (Directly enlarge a low-res image)", 
     "🔬 Scientific Evaluation (Destroy & Reconstruct a hi-res image)"],
    help="Choose whether you want to actually upscale a small picture, or mathematically simulate destruction on a 2K picture to benchmark algorithm accuracy."
)

st.sidebar.markdown("---")
st.sidebar.header("⚙ Processing Algorithms")
upscale_method = st.sidebar.radio(
    "Select Upscaling Engine:", 
    ["Lanczos-4 Interpolation (Advanced IVP)", "Mathematical Bicubic"]
)

num_cols = st.sidebar.columns([1])
scale_factor = st.sidebar.slider(
    "Select Upscaling Multiplier (Zoom Factor):", 
    min_value=2, max_value=8, value=4, step=1,
    help="Higher numbers mean larger outputs, drastically increasing dimensions."
)

st.sidebar.markdown("---")
st.sidebar.subheader("✨ Post-Process Clarity")
clarity_strength = st.sidebar.slider(
    "Unblur Intensity (Edge Sharpness)", 
    min_value=0.0, max_value=5.0, value=1.5, step=0.1,
    help="Higher values forcefully strip away blurriness and reconstruct sharp textures in real-time. Set to 0 to disable."
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
# Fixing the empty label warning
uploaded_file = st.file_uploader("Select Image File (.png, .jpg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Safely digest uploaded bytes back into fully-fledged openCV representations
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    low_res = cv2.imdecode(file_bytes, 1)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
    
    height, width = low_res.shape[:2]
    
    if "Practical Upscale" in operation_mode:
        t_size = (int(width * scale_factor), int(height * scale_factor))
        st.info(f"Target Output Dimension rendering out to: **{t_size[0]} x {t_size[1]}** pixels")
    else:
        # Scientific Evaluation
        t_size = (width, height)
        eval_tiny_w = max(1, width // scale_factor)
        eval_tiny_h = max(1, height // scale_factor)
        st.info(f"Evaluation Parameters: Destroying Original Image down to **{eval_tiny_w}x{eval_tiny_h}**, then meticulously mathematically reconstructing it back up to {width}x{height} to measure structural retention.")

    if st.button("🚀 Execute Processing Engine", use_container_width=True, type="primary"):
        with st.spinner(f"Super-Res engine crunching matrix frames natively inline..."):
            
            if "Practical Upscale" in operation_mode:
                # --- Practical Upscale Mode ---
                # --- De-Blockify Processing ---
                if apply_pixel_art and block_size > 1:
                    tiny_w = max(1, width // block_size)
                    tiny_h = max(1, height // block_size)
                    work_img = cv2.resize(low_res, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA)
                    st.info(f"Analyzed structural resolution. Image block architecture minimized to: **{tiny_w}x{tiny_h} px**.")
                else:
                    work_img = low_res

                # --- Base Architecture Engine ---
                if "Lanczos" in upscale_method:
                    output = upscale_lanczos(work_img, t_size)
                else:
                    output = upscale_bicubic(work_img, t_size)
                    if clarity_strength > 5.0:
                       output = apply_sharpening(output) 
                       
                st.session_state["input_preview"] = cv2.resize(low_res, t_size, interpolation=cv2.INTER_NEAREST)
                st.session_state["eval_metrics"] = None
                
            else:
                # --- Scientific Evaluation Mode ---
                # "low_res" is completely High-Res ground truth. Destroy it down!
                destroyed_img = cv2.resize(low_res, (eval_tiny_w, eval_tiny_h), interpolation=cv2.INTER_AREA)
                
                if "Lanczos" in upscale_method:
                    output = upscale_lanczos(destroyed_img, t_size)
                else:
                    output = upscale_bicubic(destroyed_img, t_size)
                    if clarity_strength > 5.0:
                       output = apply_sharpening(output) 
                
                # --- Graphical Analytics Computation ---
                n_out = upscale_nearest(destroyed_img, t_size)
                b_out = upscale_bicubic(destroyed_img, t_size)
                l_out = upscale_lanczos(destroyed_img, t_size)
                if clarity_strength > 0:
                    l_out = clarify_image(l_out, strength=clarity_strength)

                st.session_state["eval_metrics"] = {
                    "Nearest Neighbor (Raw baseline)": compute_metrics(low_res, n_out),
                    "Generic Bicubic math": compute_metrics(low_res, b_out),
                    "Advanced IVP (Lanczos)": compute_metrics(low_res, l_out)
                }
                
                st.session_state["input_preview"] = low_res  # Visual Slider statically compares Ground Truth vs output!

            st.session_state["base_upscaled"] = output
            st.session_state["target_size"] = t_size
            st.session_state["upscale_method"] = upscale_method
            st.success("Successfully computed output matrix! You can now freely adjust the Unblur slider below.")

# -----------------------------
# LIVE UI VISUALIZATION POOL
# -----------------------------
if "base_upscaled" in st.session_state:
    st.divider()
    
    base_out = st.session_state["base_upscaled"]
    inp_preview = st.session_state["input_preview"]
    t_size = st.session_state["target_size"]
    u_method = st.session_state["upscale_method"]

    # --- Live Unblur Adjustment Matrix ---
    if clarity_strength > 0:
        final_output = clarify_image(base_out, strength=clarity_strength)
    else:
        final_output = base_out

    st.subheader("2. 🔍 Interactive Result Inspection (Before vs After)")
    st.markdown("**(Try adjusting the *Unblur Intensity* slider on the left!)**")
    
    # -- Protection Against Gigantic Protobuf Crashes (> 4K monitors) --
    # Streamlit crashes if forced to beam gigabytes of raw matrices straight into HTML iframes.
    disp_img1 = inp_preview
    disp_img2 = final_output
    
    MAX_UI_DIM = 2500
    if t_size[0] > MAX_UI_DIM or t_size[1] > MAX_UI_DIM:
        scale_down = min(MAX_UI_DIM / t_size[0], MAX_UI_DIM / t_size[1])
        safe_ui_size = (int(t_size[0] * scale_down), int(t_size[1] * scale_down))
        disp_img1 = cv2.resize(inp_preview, safe_ui_size, interpolation=cv2.INTER_AREA)
        disp_img2 = cv2.resize(final_output, safe_ui_size, interpolation=cv2.INTER_AREA)
        
        st.warning(f"⚠️ **Preview Scaled Down:** Your upscaled {t_size[0]}x{t_size[1]} image is so phenomenally large that rendering it live would crash the browser! The slider preview below has been constrained to {safe_ui_size[0]}x{safe_ui_size[1]}, but your **Download Button** still exports the full, uncompressed {t_size[0]}x{t_size[1]} masterpiece.")

    # UI Comparison Wrapper
    image_comparison(
        img1=disp_img1,
        img2=disp_img2,
        label1=f"Original Input",
        label2=f"{u_method} (Clarity: {clarity_strength})",
        width=1000 if disp_img2.shape[1] > 1000 else disp_img2.shape[1],
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )

    # --- Live Metric Graph Viewing ---
    if "eval_metrics" in st.session_state and st.session_state["eval_metrics"] is not None:
        st.divider()
        st.subheader("📈 Scientific Accuracy Benchmarks")
        st.markdown("Because we enabled analytics, our system successfully simulated extreme data loss under the hood by physically destroying the image data, then forcefully reconstructing it using the 3 absolute core methods.")
        
        metrics = st.session_state["eval_metrics"]
        df = pd.DataFrame({
            "Algorithmic Approach": list(metrics.keys()),
            "PSNR Accuracy Score (dB)": [m["PSNR"] for m in metrics.values()],
            "SSIM Structural Score": [m["SSIM"] for m in metrics.values()]
        }).set_index("Algorithmic Approach")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Peak Signal-to-Noise Ratio** *(Measures how little noise/corruption occurred. Taller bar = exceptionally cleaner image)*")
            st.bar_chart(df[["PSNR Accuracy Score (dB)"]], color="#B48B36")
        with c2:
            st.markdown("**Structural Similarity Index** *(Measures mathematically pure edge retention. Closer to 1.0 = flawlessly crisp)*")
            st.bar_chart(df[["SSIM Structural Score"]], color="#FFFFFF")

    st.divider()

    st.subheader("3. 📥 Download Finalized Export")
    st.markdown("Your image has been firmly processed. Preserve the super resolved `.png` natively onto your machine.")
    
    # Binary buffered writing
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    io_buf = io.BytesIO(buffer)
    file_name_tag = "Lanczos4_Super_Resolution" if "Lanczos" in upscale_method else "Bicubic_Interpolation"
    file_name = f"{file_name_tag}_{t_size[0]}x{t_size[1]}_C{clarity_strength}.png"

    st.download_button(
        label=f"⬇ Download Perfected PNG (Resolution: {t_size[0]}x{t_size[1]})",
        data=io_buf,
        file_name=file_name,
        mime="image/png",
        type="primary",
        use_container_width=True
    )
else:
    if uploaded_file is None:
        st.markdown("Waiting for your image...")
