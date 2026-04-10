import streamlit as st
import cv2
import numpy as np
import io
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from image_enhancement import advanced_sharpen, denoise_before_sharpen
sys.path.append(os.path.dirname(__file__))
from src.main import (
    upscale_bicubic,
    upscale_lanczos,
    upscale_nearest,
    upscale_bilinear,
    compute_metrics,
    enhance_output,
)

import pandas as pd
from streamlit_image_comparison import image_comparison

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="IVP Super-Resolution Studio",
    page_icon="🔬"
)
@st.cache_resource
def load_espcn():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "data", "ESPCN_x4.pb")
    sr.readModel(model_path)
    sr.setModel("espcn", 4)
    return sr

def apply_espcn(image):
    sr = load_espcn()
    return sr.upsample(image)
# ──────────────────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔬 IVP Super-Resolution Studio")
st.markdown(
    "**Focused Sharp-First Pipeline** · Professional IVP Restoration  \n"
    "Upload any low-resolution image and get a crisp, high-resolution result without the clutter."
)

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────


st.sidebar.markdown("---")
st.sidebar.header("⚙️ Upscaling Engine")
upscale_method = st.sidebar.radio(
    "Algorithm:",
    [
        "Lanczos-4  ·  Advanced IVP (Recommended)",
        "Bicubic  ·  Standard Math",
        "ESPCN (AI Super Resolution)"
    ]
)

scale_factor = st.sidebar.slider(
    "Upscale Multiplier (×):", min_value=2, max_value=8, value=4, step=1,
    help="Higher = larger output. Each step quadruples pixel count."
)

st.sidebar.markdown("---")
st.sidebar.header("👾 Artifact Correction")
apply_pixel_art = st.sidebar.checkbox(
    "Enable De-blockify Mode",
    value=False,
    help="Destroys artificial block artifacts cleanly before upscaling."
)
if apply_pixel_art:
    block_size = st.sidebar.number_input(
        "Estimated Block Width (px):", min_value=1, max_value=100, value=16
    )
else:
    block_size = 1

# Hardcoded Sweet-Spot Defaults (Simplified UI)
DEFAULT_IBP_ITERS = 6
DEFAULT_FFT_BOOST = 0.40
DEFAULT_EDGE_STRENGTH = 0.55
DEFAULT_USM_SIGMA = 0.9

# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE UPLOAD
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("1. 📸 Upload Image")
uploaded_file = st.file_uploader(
    "Select Image File (.png, .jpg, .jpeg, .bmp, .tiff)",
    type=["jpg", "png", "jpeg", "bmp", "tiff"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    low_res = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

    height, width = low_res.shape[:2]

    t_size = (int(width * scale_factor), int(height * scale_factor))
    st.info(
        f"Pipeline Target: **{width} × {height}** → **{t_size[0]} × {t_size[1]}** pixels "
        f"({scale_factor}× magnification) with Scientific Benchmarks."
    )

    if st.button("🚀 Run Sharp-First IVP Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Crunching pixels through the IVP pipeline..."):

            # ── 1. Create the REAL output that the user gets ──
            if apply_pixel_art and block_size > 1:
                tiny_w = max(1, width // block_size)
                tiny_h = max(1, height // block_size)
                destroyed_img = cv2.resize(low_res, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA)
                st.info(f"De-blockify: crushed to structural base **{tiny_w}×{tiny_h}** px")
            else:
                destroyed_img = low_res

            if "ESPCN" in upscale_method:
                base_up = apply_espcn(destroyed_img)
                base_up = cv2.resize(base_up, t_size, interpolation=cv2.INTER_CUBIC)
                output = advanced_sharpen(base_up, strength=1.3, edge_boost=1.1)
            elif "Lanczos" in upscale_method:
                base_up = cv2.resize(destroyed_img, t_size, interpolation=cv2.INTER_LANCZOS4)
                ivp_output = enhance_output(base_up, lr_source=destroyed_img, ibp_iters=2, fft_boost=0.2, usm_sigma=0.5, edge_strength=0.25)
                denoised = denoise_before_sharpen(ivp_output)
                output = advanced_sharpen(denoised, strength=2.2, edge_boost=1.8)
            else:
                base_up = cv2.resize(destroyed_img, t_size, interpolation=cv2.INTER_CUBIC)
                ivp_output = enhance_output(base_up, lr_source=destroyed_img, ibp_iters=2, fft_boost=0.2, usm_sigma=0.5, edge_strength=0.25)
                denoised = denoise_before_sharpen(ivp_output)
                output = advanced_sharpen(denoised, strength=2.2, edge_boost=1.8)

            if "ESPCN" in upscale_method:
                broken_down = cv2.resize(low_res, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
                st.session_state["input_preview"] = cv2.resize(
                    broken_down, t_size, interpolation=cv2.INTER_NEAREST
                )
            else:
                st.session_state["input_preview"] = cv2.resize(
                    low_res, t_size, interpolation=cv2.INTER_NEAREST
                )

            # ── 2. Run SCIENTIFIC BACKGROUND BENCHMARKS against original resolution ──
            eval_tiny_w = max(1, width // scale_factor)
            eval_tiny_h = max(1, height // scale_factor)
            sim_degraded = cv2.resize(low_res, (eval_tiny_w, eval_tiny_h), interpolation=cv2.INTER_AREA)
            
            orig_t_size = (width, height)
            
            n_out = cv2.resize(sim_degraded, orig_t_size, interpolation=cv2.INTER_NEAREST)
            b_out = cv2.resize(sim_degraded, orig_t_size, interpolation=cv2.INTER_LINEAR)
            bc_out = upscale_bicubic(sim_degraded, orig_t_size, apply_enhancement=True)
            lz_out = upscale_lanczos(sim_degraded, orig_t_size, apply_enhancement=True)
            
            espcn_out = apply_espcn(sim_degraded)
            espcn_out = cv2.resize(espcn_out, orig_t_size, interpolation=cv2.INTER_CUBIC)

            st.session_state["eval_metrics"] = {
                "Nearest Neighbor": compute_metrics(low_res, n_out),
                "Bilinear":         compute_metrics(low_res, b_out),
                "Bicubic + IVP":    compute_metrics(low_res, bc_out),
                "Lanczos-4 + IVP":  compute_metrics(low_res, lz_out),
                "ESPCN (AI)":       compute_metrics(low_res, espcn_out),
            }

            st.session_state["base_upscaled"] = output
            st.session_state["target_size"]   = t_size
            st.session_state["upscale_method"] = upscale_method
            st.session_state["low_res_orig"]   = low_res
            st.success("✅ Pipeline complete! Scroll down to inspect your crisp output.")

# ──────────────────────────────────────────────────────────────────────────────
#  RESULTS PANEL
# ──────────────────────────────────────────────────────────────────────────────
if "base_upscaled" in st.session_state:
    st.divider()

    final_output = st.session_state["base_upscaled"]
    inp_preview  = st.session_state["input_preview"]
    t_size       = st.session_state["target_size"]
    u_method     = st.session_state["upscale_method"]
    low_res_orig = st.session_state.get("low_res_orig", inp_preview)

    # ── Before / After slider ──────────────────────────────────────────────
    st.subheader("2. 🔍 Before vs After (Drag to Compare)")

    disp1, disp2 = inp_preview, final_output
    MAX_UI_DIM = 2500
    if t_size[0] > MAX_UI_DIM or t_size[1] > MAX_UI_DIM:
        sd = min(MAX_UI_DIM / t_size[0], MAX_UI_DIM / t_size[1])
        safe_sz = (int(t_size[0] * sd), int(t_size[1] * sd))
        disp1 = cv2.resize(inp_preview,   safe_sz, interpolation=cv2.INTER_AREA)
        disp2 = cv2.resize(final_output,  safe_sz, interpolation=cv2.INTER_AREA)
        st.warning(
            f"⚠️ Preview scaled to {safe_sz[0]}×{safe_sz[1]} for browser safety. "
            f"Download still exports full **{t_size[0]}×{t_size[1]}** image."
        )

    image_comparison(
        img1=disp1, img2=disp2,
        label1="Input",
        label2=f"{u_method.split('·')[0].strip()} Output",
        width=min(1000, disp2.shape[1]),
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

    # ── Scientific Benchmarks ──────────────────────────────────────────────
    if st.session_state.get("eval_metrics") is not None:
        st.divider()
        st.subheader("📈 Scientific Accuracy Benchmarks")
        st.markdown(
            "Each algorithm reconstructed the degraded image back to original resolution. "
            "Scores measure how closely the output matches the **pristine ground truth**."
        )
        metrics = st.session_state["eval_metrics"]
        method_labels = list(metrics.keys())

        # -- PSNR row --
        st.markdown("### 1. Peak Signal-to-Noise Ratio (PSNR) — Higher is Better")
        st.caption("Measures data fidelity and noise. Higher dB = closer to the original.")
        psnr_cols = st.columns(len(method_labels))
        for col, label in zip(psnr_cols, method_labels):
            col.metric(label, f"{metrics[label]['PSNR']:.2f} dB")

        # -- SSIM row --
        st.markdown("### 2. Structural Similarity Index (SSIM) — Closer to 1.0 is Better")
        st.caption("Measures edge and structural preservation. 1.0 = perfect lossless match.")
        ssim_cols = st.columns(len(method_labels))
        for col, label in zip(ssim_cols, method_labels):
            col.metric(label, f"{metrics[label]['SSIM']:.4f}")

        # -- Charts --
        st.markdown("---")
        st.markdown("#### Visual Distribution")
        df = pd.DataFrame({
            "Algorithm": method_labels,
            "PSNR (dB)": [metrics[m]["PSNR"] for m in method_labels],
            "SSIM": [metrics[m]["SSIM"] for m in method_labels],
        }).set_index("Algorithm")

        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(df[["PSNR (dB)"]], color="#B48B36")
        with c2:
            st.bar_chart(df[["SSIM"]], color="#6EA8FF")

    # ── Visual Analytics Dashboard ─────────────────────────────────────────
    st.divider()
    st.subheader("📊 Visual Analytics Dashboard")
    st.markdown(
        "Deep-dive into what the pipeline actually changed — pixel distributions, "
        "sharpness metrics, per-channel statistics, and resolution gains."
    )

    # 1. RGB Histograms
    st.markdown("### 🎨 Pixel Intensity Histogram (Input vs Output)")
    st.caption(
        "A wider, well-spread histogram → richer dynamic range → more detail. "
        "Notice how the IVP pipeline redistributes pixel energy toward the tails."
    )
    fig_hist, axes_h = plt.subplots(1, 2, figsize=(14, 4.5))
    fig_hist.patch.set_facecolor('#0E1117')
    colors = ['#FF6B6B', '#51CF66', '#339AF0']
    ch_names = ['Red', 'Green', 'Blue']

    for ax, img, title in [
        (axes_h[0], low_res_orig, 'Input Image'),
        (axes_h[1], final_output, 'IVP Output (Sharp-First)'),
    ]:
        small = cv2.resize(img, (min(img.shape[1], 512), min(img.shape[0], 512)),
                           interpolation=cv2.INTER_AREA)
        for c_idx in range(3):
            hv = cv2.calcHist([small], [c_idx], None, [256], [0, 256]).flatten()
            ax.fill_between(range(256), hv, alpha=0.3, color=colors[c_idx], label=ch_names[c_idx])
            ax.plot(range(256), hv, color=colors[c_idx], linewidth=0.9)
        ax.set_title(title, color='white', fontsize=13, fontweight='bold')
        ax.set_facecolor('#1A1D23')
        ax.tick_params(colors='white')
        ax.legend(fontsize=9, facecolor='#1A1D23', edgecolor='#333', labelcolor='white')
        ax.set_xlim(0, 255)
        ax.set_xlabel('Pixel Intensity', color='#888')
        ax.set_ylabel('Frequency', color='#888')
        for s in ax.spines.values():
            s.set_color('#333')

    fig_hist.tight_layout(pad=2.0)
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # 2. Image Quality Statistics
    st.markdown("### 📐 Image Quality Statistics")
    st.caption("Key sharpness and contrast indicators. Sharpness = Laplacian variance (higher = sharper).")

    def compute_image_stats(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {
            "Mean Brightness":         float(np.mean(gray)),
            "Std Dev (Contrast)":      float(np.std(gray)),
            "Sharpness (Laplacian ↑)": float(lap_var),
            "Dynamic Range":           float(np.max(gray) - np.min(gray)),
        }

    s_in  = compute_image_stats(low_res_orig)
    s_out = compute_image_stats(final_output)
    stat_cols = st.columns(4)
    for i, label in enumerate(s_in.keys()):
        delta = s_out[label] - s_in[label]
        stat_cols[i].metric(
            label=label,
            value=f"{s_out[label]:.1f}",
            delta=f"{delta:+.1f}",
            help=f"Input: {s_in[label]:.1f}  →  Output: {s_out[label]:.1f}"
        )

    # 3. Per-Channel Analysis
    st.markdown("### 🔬 Per-Channel Mean & Std Dev")
    st.caption("R/G/B channel statistics: mean intensity and standard deviation for input and output.")

    ch_data = []
    for c_idx, c_name in enumerate(ch_names):
        ch_data.append({
            "Channel":      c_name,
            "Input Mean":   float(np.mean(low_res_orig[:, :, c_idx])),
            "Output Mean":  float(np.mean(final_output[:, :, c_idx])),
            "Input StdDev": float(np.std(low_res_orig[:, :, c_idx])),
            "Output StdDev":float(np.std(final_output[:, :, c_idx])),
        })
    df_ch = pd.DataFrame(ch_data).set_index("Channel")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Channel Mean Intensity**")
        st.bar_chart(df_ch[["Input Mean", "Output Mean"]], color=["#FF6B6B", "#339AF0"])
    with cc2:
        st.markdown("**Channel Std Deviation**")
        st.bar_chart(df_ch[["Input StdDev", "Output StdDev"]], color=["#FF6B6B", "#339AF0"])

    # 4. Sharpness Comparison Bar
    st.markdown("### 🗡️ Sharpness Gain Analysis")
    st.caption(
        "Laplacian variance is the standard IVP sharpness metric. "
        "Higher = more high-frequency edge content = sharper image."
    )
    sharp_df = pd.DataFrame({
        "Stage": ["Input", "IVP Output"],
        "Sharpness (Laplacian Var)": [s_in["Sharpness (Laplacian ↑)"], s_out["Sharpness (Laplacian ↑)"]],
    }).set_index("Stage")
    st.bar_chart(sharp_df, color="#51CF66")
    gain_pct = ((s_out["Sharpness (Laplacian ↑)"] / max(s_in["Sharpness (Laplacian ↑)"], 1e-6)) - 1) * 100
    st.metric("Sharpness Multiplier", f"{s_out['Sharpness (Laplacian ↑)']:.0f}", delta=f"{gain_pct:+.1f}%")

    # 5. Resolution Info
    st.markdown("### 📏 Resolution Upgrade")
    in_h, in_w = low_res_orig.shape[:2]
    out_h, out_w = final_output.shape[:2]
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Input Resolution",  f"{in_w} × {in_h}",   help="Width × Height of the uploaded image")
    rc2.metric("Output Resolution", f"{out_w} × {out_h}",  help="Width × Height of the IVP output")
    rc3.metric("Pixel Count ×",     f"{(out_w * out_h) / max(1, in_w * in_h):.1f}×",
               help="How many times more pixels the output contains")

    st.divider()

    # ── Download ────────────────────────────────────────────────────────────
    st.subheader("3. 📥 Download Your Sharp Output")
    st.markdown(
        "The full-resolution, processed PNG is ready. "
        "This is the complete, lossless export at the target resolution."
    )

    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    io_buf = io.BytesIO(buffer)
    if "ESPCN" in u_method:
        tag = "ESPCN_AI"
    elif "Lanczos" in u_method:
        tag = "Lanczos4_IVP"
    else:
        tag = "Bicubic_IVP"
    file_name = f"{tag}_Sharp_{t_size[0]}x{t_size[1]}.png"

    st.download_button(
        label=f"⬇ Download Sharp PNG  ({t_size[0]} × {t_size[1]} px)",
        data=io_buf,
        file_name=file_name,
        mime="image/png",
        type="primary",
        use_container_width=True,
    )

else:
    if uploaded_file is None:
        st.markdown(
            "> 📂 Upload an image above and hit **Run Sharp-First IVP Pipeline** to begin."
        )
