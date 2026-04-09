import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import os

# ═══════════════════════════════════════════════════════════════════════════════
#  IVP CORE LIBRARY  ·  Sharp-First Super-Resolution Pipeline
#  Concepts: Lanczos-4, Iterative Back-Projection, Frequency-Domain Sharpening,
#            Edge-Directed Adaptive USM, Anisotropic Edge Enhancement
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(filepath):
    """Load an image from disk and convert BGR → RGB."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {filepath}.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def downscale_image(image, scale_factor=0.25):
    """
    Simulate low-resolution via INTER_AREA decimation.
    Only used for evaluation/benchmarking — not in production upscale flow.
    """
    h, w = image.shape[:2]
    nw = max(1, int(w * scale_factor))
    nh = max(1, int(h * scale_factor))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE A · Frequency-Domain High-Pass Detail Injection (FFT Sharpening)
#
#  Converts the image to the frequency domain via DFT, amplifies all
#  high-frequency components (edges, fine texture, detail) using a
#  Gaussian high-pass mask, and merges the restored detail back into
#  the spatial image.  Crucially: zero smoothing involved.
# ─────────────────────────────────────────────────────────────────────────────
def _fft_highpass_sharpen(image: np.ndarray, boost: float = 0.45, cutoff_ratio: float = 0.06) -> np.ndarray:
    """
    FFT-based high-frequency detail booster.

    Parameters
    ----------
    image       : float32 RGB image [0,255]
    boost       : how strongly to blend the recovered detail (0 = none, 1 = full)
    cutoff_ratio: fraction of image size below which freq components are suppressed
                  (smaller = sharper, more detail recovered)

    Returns
    -------
    Sharpened float32 RGB image, clipped to [0, 255].
    """
    result = image.copy()
    h, w = image.shape[:2]

    # Build frequency-domain Gaussian high-pass mask  (1 – Gaussian_low)
    cy, cx = h // 2, w // 2
    sigma = cutoff_ratio * min(h, w)
    yy, xx = np.ogrid[:h, :w]
    gaussian_lp = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    hp_mask = 1.0 - gaussian_lp   # high-pass: attenuate DC, pass edges

    for c in range(3):
        channel = image[:, :, c]
        # DFT → shift DC to centre
        dft = np.fft.fftshift(np.fft.fft2(channel))
        # Isolate high-frequency detail layer
        hp_detail = np.real(np.fft.ifft2(np.fft.ifftshift(dft * hp_mask)))
        # Add back the detail layer with `boost` weight
        result[:, :, c] = np.clip(channel + boost * hp_detail, 0, 255)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE B · Iterative Back-Projection (IBP) Refinement
#
#  Classic IVP super-resolution post-processor.
#  Algorithm:
#    1. Degrade the current estimate H back to LR space (INTER_AREA)
#    2. Compute residual error: LR_original − LR_from_H
#    3. Upscale the residual error to HR space
#    4. Add the projected error back to H
#    5. Repeat for `iterations` steps
#
#  Effect: iteratively corrects interpolation artefacts so the output is
#  self-consistent with the known LR source — no blurring involved.
# ─────────────────────────────────────────────────────────────────────────────
def iterative_back_projection(lr_image: np.ndarray,
                               hr_image: np.ndarray,
                               iterations: int = 6,
                               step_size: float = 0.35) -> np.ndarray:
    """
    Iterative Back-Projection refinement.

    Parameters
    ----------
    lr_image   : original low-resolution source (uint8 or float32)
    hr_image   : initial high-resolution estimate (uint8 or float32)
    iterations : number of projection cycles (more = more faithful to LR source)
    step_size  : learning rate per iteration (0.1 – 0.6 recommended)

    Returns
    -------
    Refined HR image as uint8.
    """
    lr_h, lr_w = lr_image.shape[:2]
    hr_h, hr_w = hr_image.shape[:2]

    hr = hr_image.astype(np.float32)
    lr_ref = lr_image.astype(np.float32)

    for _ in range(iterations):
        # Step 1: Degrade HR estimate → LR space
        hr_degraded = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        # Step 2: Pixel-level residual error in LR domain
        error_lr = lr_ref - hr_degraded

        # Step 3: Upscale error map to HR space (Lanczos for accuracy)
        error_hr = cv2.resize(error_lr, (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4)

        # Step 4: Project error back into HR estimate
        hr = hr + step_size * error_hr
        hr = np.clip(hr, 0, 255)

    return hr.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE C · Edge-Directed Anisotropic Enhancement
#
#  Computes the Scharr gradient magnitude map (more isotropic than Sobel).
#  Edges with strong gradients receive targeted sharpening via a local
#  high-pass kernel.  Flat/smooth regions are left completely untouched —
#  preventing noise amplification while maximally sharpening structure.
# ─────────────────────────────────────────────────────────────────────────────
def _edge_directed_sharpen(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Anisotropic edge-directed sharpening.

    Applies a high-pass sharpening kernel *only* where the gradient
    magnitude is high (detected edges), leaving flat regions untouched.

    Parameters
    ----------
    image    : uint8 RGB image
    strength : blend ratio for the sharpened layer at detected edges

    Returns
    -------
    Sharpened uint8 RGB image.
    """
    img_f = image.astype(np.float32)
    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Scharr gradients — more precise than Sobel for fine edges
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Normalise gradient map to [0, 1] — this becomes the blending mask
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    edge_mask = grad_norm[:, :, np.newaxis]   # broadcast over channels

    # Unsharpen: tiny sigma = high-frequency focus, no blob blurring
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=0.6)
    detail  = img_f - blurred

    # Apply enhancement only at detected edge locations
    sharpened = img_f + strength * detail * edge_mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE D · Adaptive Variance-Weighted Unsharp Masking
#
#  Unlike fixed unsharp masking, this computes the local variance of the
#  image in a small neighbourhood.  Pixels in high-variance (texture-rich)
#  regions receive strong sharpening; pixels in low-variance (smooth) regions
#  receive almost no sharpening — perfectly balancing sharpness vs noise.
# ─────────────────────────────────────────────────────────────────────────────
def _adaptive_usm(image: np.ndarray,
                  sigma: float = 1.0,
                  max_strength: float = 1.2,
                  var_window: int = 7) -> np.ndarray:
    """
    Adaptive Unsharp Masking (variance-weighted).

    Parameters
    ----------
    image        : uint8 RGB image
    sigma        : Gaussian blur radius for extracting the detail layer
    max_strength : maximum sharpening amplification
    var_window   : local window size for variance estimation

    Returns
    -------
    Sharpened uint8 RGB image.
    """
    img_f  = image.astype(np.float32)
    blur   = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    detail = img_f - blur

    # Local variance map (proxy for texture richness)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu    = cv2.blur(img_gray, (var_window, var_window))
    mu2   = cv2.blur(img_gray**2, (var_window, var_window))
    var_map = np.maximum(mu2 - mu**2, 0)

    # Normalise: high variance → strong boost, low variance → weak boost
    var_norm = var_map / (var_map.max() + 1e-6)
    strength_map = (var_norm * max_strength)[:, :, np.newaxis]

    result = img_f + strength_map * detail
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER ENHANCEMENT PIPELINE  (no blur ever applied globally)
# ─────────────────────────────────────────────────────────────────────────────
def enhance_output(image: np.ndarray,
                   lr_source: np.ndarray = None,
                   ibp_iters: int = 6,
                   fft_boost: float = 0.40,
                   usm_sigma: float = 0.9,
                   edge_strength: float = 0.55) -> np.ndarray:
    """
    Four-stage Sharp-First IVP Enhancement Pipeline.

    Stage 1 — Iterative Back-Projection (IBP)
        IVP classic: iteratively corrects the HR estimate so it is
        self-consistent with the LR source. Removes ring/ringing artefacts
        introduced by interpolation kernels with zero blur.

    Stage 2 — FFT High-Pass Detail Injection
        Recovers edge and texture frequencies lost during upscaling by
        amplifying high-frequency DFT components.  All in frequency space.

    Stage 3 — Edge-Directed Anisotropic Sharpening
        Scharr-gradient-gated high-pass sharpening: only edges are
        sharpened, so noise is not amplified in flat areas.

    Stage 4 — Adaptive Variance-Weighted Unsharp Masking
        Texture-aware USM: high-frequency detail amplified proportionally
        to local image variance. Zero smoothing applied.

    Parameters
    ----------
    image        : initial upscaled HR image (uint8 RGB)
    lr_source    : original LR input (for IBP correction); None skips IBP
    ibp_iters    : IBP iteration count (5–8 recommended)
    fft_boost    : FFT high-pass blend strength (0.3–0.6)
    usm_sigma    : USM Gaussian sigma (0.7–1.2)
    edge_strength: anisotropic edge sharpening weight (0.4–0.7)
    """
    result = image.copy()

    # ── Stage 1: Iterative Back-Projection ──────────────────────────────────
    if lr_source is not None:
        result = iterative_back_projection(lr_source, result, iterations=ibp_iters)

    # ── Stage 2: FFT High-Pass Detail Injection ──────────────────────────────
    result_f = result.astype(np.float32)
    result_f = _fft_highpass_sharpen(result_f, boost=fft_boost, cutoff_ratio=0.07)
    result = np.clip(result_f, 0, 255).astype(np.uint8)

    # ── Stage 3: Edge-Directed Anisotropic Sharpening ───────────────────────
    result = _edge_directed_sharpen(result, strength=edge_strength)

    # ── Stage 4: Adaptive Variance-Weighted Unsharp Masking ─────────────────
    result = _adaptive_usm(result, sigma=usm_sigma, max_strength=1.1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC UPSCALE METHODS
# ─────────────────────────────────────────────────────────────────────────────

def upscale_lanczos(image: np.ndarray, target_size: tuple,
                    apply_enhancement: bool = True) -> np.ndarray:
    """
    Lanczos-4 upscaling + full Sharp-First IVP enhancement pipeline.
    Best quality: uses an 8×8 sinc-windowed kernel, then corrects via IBP.
    """
    upscaled = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    if apply_enhancement:
        return enhance_output(upscaled, lr_source=image)
    return upscaled


def upscale_bicubic(image: np.ndarray, target_size: tuple,
                    apply_enhancement: bool = True) -> np.ndarray:
    """
    Bicubic upscaling + full Sharp-First IVP enhancement pipeline.
    """
    upscaled = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    if apply_enhancement:
        return enhance_output(upscaled, lr_source=image)
    return upscaled


def upscale_bilinear(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Bilinear interpolation (baseline, no enhancement)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def upscale_nearest(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Nearest-neighbour interpolation (raw baseline)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────────────────────
#  LEGACY / UTILITY  (kept for CLI pipeline compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """Classic 3×3 Laplacian sharpening kernel (legacy support)."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def clarify_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Legacy-compatible wrapper now backed by adaptive USM (not blurring).
    strength maps to max_strength parameter of _adaptive_usm.
    """
    if strength <= 0:
        return image
    return _adaptive_usm(image, sigma=0.8, max_strength=min(strength, 2.0))


def compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """PSNR + SSIM quality metrics vs. ground-truth original."""
    if original.shape != processed.shape:
        processed = cv2.resize(processed,
                               (original.shape[1], original.shape[0]),
                               interpolation=cv2.INTER_LANCZOS4)
    psnr_val = psnr_metric(original, processed)
    ssim_val = ssim_metric(original, processed, channel_axis=2, data_range=255)
    return {"PSNR": psnr_val, "SSIM": ssim_val}


def print_metrics(results: dict):
    print("\n--- Evaluation Metrics ---")
    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
    print("--------------------------\n")


def create_grid_figure(images_dict: dict):
    cols = 3
    rows = (len(images_dict) + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5 * rows))
    for i, (title, img) in enumerate(images_dict.items()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=14)
        plt.axis('off')
    fig.tight_layout()
    return fig


def create_chart_figure(results: dict):
    methods = list(results.keys())
    psnr_scores = [results[m]["PSNR"] for m in methods]
    ssim_scores = [results[m]["SSIM"] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('PSNR (dB)', color='tab:blue')
    bars1 = ax1.bar(x - width/2, psnr_scores, width, label='PSNR', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('SSIM', color='tab:orange')
    bars2 = ax2.bar(x + width/2, ssim_scores, width, label='SSIM', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    plt.title('Performance Comparison: PSNR vs SSIM')

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1,
                 f'{yval:.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005,
                 f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("   IVP SUPER-RESOLUTION PIPELINE  ·  Sharp-First Edition")
    print("="*60 + "\n")

    input_path = input("1) Image path to upscale: ").strip().strip('"').strip("'")
    if not input_path:
        print("Error: No image path provided. Exiting.")
        return

    try:
        input_img = load_image(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\nSelect Operation Mode:")
    print("  [1] Evaluation (downscale → upscale → measure vs original)")
    print("  [2] Practical Upscale (directly enlarge a low-res image)")
    print("  [3] De-Blockify / Pixel-Art Mode")
    mode_choice = input("Enter 1, 2, or 3 [Default 1]: ").strip()

    if mode_choice == "3":
        mode = "pixel"
    elif mode_choice == "2":
        mode = "upscale"
    else:
        mode = "eval"

    h, w = input_img.shape[:2]

    if mode == "eval":
        original_img = input_img
        scale_str = input("\nDownscale factor (e.g. 0.25) [Default 0.25]: ").strip()
        scale = float(scale_str) if scale_str else 0.25
        target_size = (w, h)
        print(f"\nOriginal: {w}×{h}")
        low_res_img = downscale_image(original_img, scale)
        lrh, lrw = low_res_img.shape[:2]
        print(f"Simulated LR: {lrw}×{lrh}")

    elif mode == "pixel":
        block_size_str = input("\nEstimated block size in px (e.g. 10) [Default 15]: ").strip()
        block_size = int(block_size_str) if block_size_str else 15
        scale_str = input("Final upscale multiplier (e.g. 2.0) [Default 2.0]: ").strip()
        scale = float(scale_str) if scale_str else 2.0
        target_size = (int(w * scale), int(h * scale))
        tiny_w = max(1, w // block_size)
        tiny_h = max(1, h // block_size)
        low_res_img = cv2.resize(input_img, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA)
        print(f"\nBlocky input: {w}×{h}")
        print(f"Crushed to structural pixels: {tiny_w}×{tiny_h}")
        print(f"Target upscaled: {target_size[0]}×{target_size[1]}")

    else:
        low_res_img = input_img
        scale_str = input("\nUpscale multiplier (e.g. 2.0) [Default 2.0]: ").strip()
        scale = float(scale_str) if scale_str else 2.0
        target_size = (int(w * scale), int(h * scale))
        print(f"\nInput: {w}×{h}")
        print(f"Target: {target_size[0]}×{target_size[1]}")

    print("\n[+] Running Sharp-First IVP pipeline...")
    print("    · Lanczos-4 → IBP → FFT high-pass → Edge USM → Adaptive USM")

    bicubic_out   = upscale_bicubic(low_res_img, target_size)
    bilinear_out  = upscale_bilinear(low_res_img, target_size)
    nearest_out   = upscale_nearest(low_res_img, target_size)
    lanczos_out   = upscale_lanczos(low_res_img, target_size)

    # Best output = Lanczos + full pipeline
    best_out = lanczos_out

    results = {}
    if mode == "eval":
        print("[+] Computing PSNR & SSIM metrics...")
        results = {
            "Nearest Neighbor": compute_metrics(original_img, nearest_out),
            "Bilinear":         compute_metrics(original_img, bilinear_out),
            "Bicubic + IVP":    compute_metrics(original_img, bicubic_out),
            "Lanczos-4 + IVP":  compute_metrics(original_img, lanczos_out),
        }
        print_metrics(results)

    images_to_display = {}
    if mode == "eval":
        images_to_display["Original (Reference)"]  = original_img
        images_to_display["Simulated LR"]           = low_res_img
    elif mode == "pixel":
        images_to_display["Input (Blocky)"]         = input_img
        images_to_display["Structural Pixels"]      = low_res_img
    else:
        images_to_display["Input LR"]               = low_res_img

    images_to_display["Nearest Neighbor"]           = nearest_out
    images_to_display["Bilinear"]                   = bilinear_out
    images_to_display["Bicubic + IVP Pipeline"]     = bicubic_out
    images_to_display["Lanczos-4 + IVP Pipeline"]   = lanczos_out

    print("\n[*] Displaying comparison grid...")
    print("    --> Close the window to continue <--")
    grid_fig = create_grid_figure(images_to_display)
    plt.show()

    if mode == "eval" and results:
        print("[*] Displaying performance charts...")
        chart_fig = create_chart_figure(results)
        plt.show()

    print("\n" + "="*60)
    out_dir = input("2) Output directory [Default: ./output]: ").strip()
    if not out_dir:
        out_dir = "./output"

    os.makedirs(out_dir, exist_ok=True)

    # Save final best output
    final_path = os.path.join(out_dir, "SUPER_RESOLVED_FINAL_OUTPUT.png")
    cv2.imwrite(final_path, cv2.cvtColor(best_out, cv2.COLOR_RGB2BGR))
    print(f"    -> [🏆] Final sharp result: {final_path}")

    # Save comparison grid
    grid_path = os.path.join(out_dir, "results_grid_collage.png")
    grid_fig_save = create_grid_figure(images_to_display)
    grid_fig_save.savefig(grid_path, bbox_inches='tight')
    plt.close(grid_fig_save)
    print(f"    -> Comparison grid saved: {grid_path}")

    if mode == "eval":
        chart_path = os.path.join(out_dir, "metrics_chart.png")
        chart_fig_save = create_chart_figure(results)
        chart_fig_save.savefig(chart_path, bbox_inches='tight')
        plt.close(chart_fig_save)
        print(f"    -> Metrics chart saved: {chart_path}")

    print("\n[✓] Done. Enjoy your crisp, super-resolved image!")


if __name__ == "__main__":
    main()
