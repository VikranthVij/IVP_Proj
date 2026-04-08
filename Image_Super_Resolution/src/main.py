import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import os

def load_image(filepath):
    """
    Load an image from the given path and convert it from BGR to RGB.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")
    
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {filepath}. Ensure it is a valid image format.")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def downscale_image(image, scale_factor=0.25):
    """
    Simulate Low Resolution.
    Note: Downscaling is used only for evaluation and not part of real-world deployment.
    Uses INTER_AREA which is preferred for image decimation.
    """
    height, width = image.shape[:2]
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    
    low_res = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return low_res

def upscale_bicubic(image, target_size):
    """
    Upscale using Bicubic Interpolation (cv2.INTER_CUBIC)
    Target size is (width, height)
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

def upscale_bilinear(image, target_size):
    """
    Upscale using Bilinear Interpolation (cv2.INTER_LINEAR)
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def upscale_nearest(image, target_size):
    """
    Upscale using Nearest Neighbor Interpolation (cv2.INTER_NEAREST)
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

def upscale_ai(image, target_size):
    """
    Upscale using Deep Learning ESPCN model to achieve real Super Resolution.
    Currently uses an x4 model and resizes to target_size seamlessly.
    """
    if not hasattr(cv2, 'dnn_superres'):
        print("\n[!] Warning: AI Super Resolution requires 'opencv-contrib-python' instead of standard 'opencv-python'.")
        print("    Please run: pip uninstall opencv-python && pip install opencv-contrib-python")
        print("    Falling back to standard Bicubic Interpolation for the 'AI' panel.\n")
        return upscale_bicubic(image, target_size)
        
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data", "ESPCN_x4.pb")
    if not os.path.exists(model_path):
        print(f"Warning: AI Model not found at {model_path}. Falling back to Bicubic.")
        return upscale_bicubic(image, target_size)
        
    sr.readModel(model_path)
    sr.setModel("espcn", 4)
    result = sr.upsample(image)
    
    if result.shape[:2] != (target_size[1], target_size[0]):
        # Match user's requested display multiplier size precisely with CUBIC instead of AREA to prevent blockiness
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_CUBIC)
    return result

def apply_sharpening(image):
    """
    Apply a harsh sharpening filter to the image to improve visual clarity for standard algorithms.
    Using a stronger kernel to make the sharpening differences more pronounced.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def clarify_image(image, strength=1.5):
    """
    Applies an infinitely scaled Unsharp Masking to naturally strip away optical blurriness 
    without generating harsh geometric artifact rings. Customizable via 'strength'.
    Formula: original + (original - blur) * strength
    """
    if strength <= 0:
        return image
        
    # Drastically increased the blur radius (sigma=15.0) to successfully target massive 
    # structural blur gradients across heavily upscaled HD images instead of microscopic pixel noise!
    blurred = cv2.GaussianBlur(image, (0, 0), 10.0)
    
    weight1 = 1.0 + strength
    weight2 = -strength
    clarified = cv2.addWeighted(image, weight1, blurred, weight2, 0)
    return clarified

def compute_metrics(original, processed):
    """
    Compare the processed image with the original using PSNR and SSIM.
    Returns a dictionary of metrics.
    """
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
        
    psnr_val = psnr_metric(original, processed)
    ssim_val = ssim_metric(original, processed, channel_axis=2, data_range=255)
    
    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val
    }

def print_metrics(results):
    """
    Print the evaluation metrics clearly.
    """
    print("\n--- Evaluation Metrics ---")
    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  PSNR: {metrics['PSNR']:.2f}")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
    print("--------------------------\n")

def create_grid_figure(images_dict):
    num_images = len(images_dict)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(15, 5 * rows))
    for i, (title, img) in enumerate(images_dict.items()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=14)
        plt.axis('off')
        
    fig.tight_layout()
    return fig

def create_chart_figure(results):
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
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.1f}', ha='center', va='bottom', color='black', fontsize=9)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', color='black', fontsize=9)

    fig.tight_layout()
    return fig

def main():
    print("\n" + "="*50)
    print("      IMAGE SUPER RESOLUTION EVALUATION PIPELINE     ")
    print("="*50 + "\n")

    input_path = input("1) Please enter the path to the picture you want to upload/process: ").strip()
    
    # Strip wrapper quotes if user dragged and dropped in terminal
    if input_path.startswith('"') and input_path.endswith('"'): input_path = input_path[1:-1]
    if input_path.startswith("'") and input_path.endswith("'"): input_path = input_path[1:-1]
    
    if not input_path:
        print("Error: No image path provided. Exiting.")
        return

    try:
        input_img = load_image(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print("\nSelect Operation Mode:")
    print("  [1] Evaluation Mode (Downscales your high-res pic, upscales, and calculates metrics against the original reference picture)")
    print("  [2] Practical Upscale (Directly upscales your picture seamlessly without downscaling)")
    print("  [3] De-Blockify / Pixel-Art Mode (Forces a massive downscale to destroy rigid block artifacts before organically upscaling)")
    mode_choice = input("Enter 1, 2, or 3 [Default 1]: ").strip()
    
    if mode_choice == "3":
        mode = "pixel"
    elif mode_choice == "2":
        mode = "upscale"
    else:
        mode = "eval"
        
    height, width = input_img.shape[:2]
    
    if mode == "eval":
        original_img = input_img
        scale_str = input("\nEnter downscale factor (e.g. 0.25) [Default 0.25]: ").strip()
        scale = float(scale_str) if scale_str else 0.25
        
        target_size = (width, height)
        print(f"\nOriginal Reference Picture size: {width}x{height}")
        low_res_img = downscale_image(original_img, scale)
        lr_height, lr_width = low_res_img.shape[:2]
        print(f"Simulated Low Resolution size: {lr_width}x{lr_height}")
    elif mode == "pixel":
        block_size_str = input("\nEnter estimated block size in pixels (e.g., 10 or 15) [Default 15]: ").strip()
        block_size = int(block_size_str) if block_size_str else 15
        
        scale_str = input("Enter final upscale multiplier (e.g. 1.0 or 2.0) [Default 2.0]: ").strip()
        scale = float(scale_str) if scale_str else 2.0
        target_size = (int(width * scale), int(height * scale))
        
        # Destroy the artificial blocks by crushing it to true resolution
        tiny_w = max(1, width // block_size)
        tiny_h = max(1, height // block_size)
        low_res_img = cv2.resize(input_img, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA)
        
        print(f"\nOriginal Blocky Picture size: {width}x{height}")
        print(f"Crushed to true structural pixel size: {tiny_w}x{tiny_h} to destroy artifacts.")
        print(f"Target Final Upscaled Size: {target_size[0]}x{target_size[1]}")
    else:
        low_res_img = input_img
        scale_str = input("\nEnter upscale multiplier (e.g. 2.0 or 4.0) [Default 2.0]: ").strip()
        scale = float(scale_str) if scale_str else 2.0
        
        target_size = (int(width * scale), int(height * scale))
        print(f"\nInput Low Resolution Picture size: {width}x{height}")
        print(f"Target Upscaled Size: {target_size[0]}x{target_size[1]}")

    print("\n[+] Generating Computations (Applying Upscaling Algorithms, Sharpening, and AI ESPCN)...")
    bicubic_out = upscale_bicubic(low_res_img, target_size)
    bilinear_out = upscale_bilinear(low_res_img, target_size)
    nearest_out = upscale_nearest(low_res_img, target_size)
    ai_out = upscale_ai(low_res_img, target_size)
    
    # Optional unblur smoothing natively implemented here for the AI variant
    ai_out = clarify_image(ai_out, strength=1.5)
    
    sharpened_bicubic_out = apply_sharpening(bicubic_out)
    
    results = {}
    if mode == "eval":
        print("[+] Calculating PSNR & SSIM Evaluation Metrics...")
        results = {
            "Bicubic": compute_metrics(original_img, bicubic_out),
            "Sharpened Bicubic": compute_metrics(original_img, sharpened_bicubic_out),
            "Bilinear": compute_metrics(original_img, bilinear_out),
            "Nearest Neighbor": compute_metrics(original_img, nearest_out),
            "AI (ESPCN)": compute_metrics(original_img, ai_out)
        }
        print_metrics(results)
    
    images_to_display = {}
    if mode == "eval":
        images_to_display["Original Image (Reference)"] = original_img
        images_to_display["Low Resolution (Simulated)"] = low_res_img
    elif mode == "pixel":
        images_to_display["Input (Blocky)"] = input_img
        images_to_display["Crushed Structural View"] = low_res_img
    else:
        images_to_display["Input Image (Low Res)"] = low_res_img

    images_to_display["Nearest Neighbor"] = nearest_out
    images_to_display["Bilinear Output"] = bilinear_out
    images_to_display["Bicubic Output"] = bicubic_out
    images_to_display["Sharpened Bicubic"] = sharpened_bicubic_out
    images_to_display["AI Super Res (ESPCN)"] = ai_out
    
    print("\n[*] Computations complete! Displaying the comparison pictures as a single image.")
    print("    --> IMPORTANT: Close the image window when you are done to continue the terminal prompt. <--")
    
    # Display the grid window interactively
    grid_fig = create_grid_figure(images_to_display)
    plt.show() 

    if mode == "eval" and len(results) > 0:
        print("\n[*] Displaying Performance Charts.")
        print("    --> IMPORTANT: Close the chart window when you are done to continue. <--")
        chart_fig = create_chart_figure(results)
        plt.show()
    
    print("\n" + "="*50)
    out_dir = input("2) Where would you like to download/save the full-scale individual pictures and comparison grid? \n   Enter output directory path (e.g. './output' or 'C:/Downloads') [Default: ./output]: ").strip()
    if not out_dir:
        out_dir = "./output"
        
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[+] Saving generated artifacts to '{out_dir}/' ...")
    
    # Save the absolute definitive final result completely independently and clearly
    final_best_output = ai_out 
    final_output_path = os.path.join(out_dir, "SUPER_RESOLVED_FINAL_OUTPUT.png")
    cv2.imwrite(final_output_path, cv2.cvtColor(final_best_output, cv2.COLOR_RGB2BGR))
    print(f"    -> [🏆] Saved Your Awesome Final Upscaled Picture to: {final_output_path}")

    # Save the grid
    grid_save_path = os.path.join(out_dir, "results_grid_collage.png")
    grid_fig_save = create_grid_figure(images_to_display)
    grid_fig_save.savefig(grid_save_path, bbox_inches='tight')
    plt.close(grid_fig_save)
    print(f"    -> Saved Comparison Grid Collage to {grid_save_path}")
    
    if mode == "eval":
        chart_save_path = os.path.join(out_dir, "metrics_chart.png")
        chart_fig_save = create_chart_figure(results)
        chart_fig_save.savefig(chart_save_path, bbox_inches='tight')
        plt.close(chart_fig_save)
        print(f"    -> Saved Metrics Chart to {chart_save_path}")

    print("\n[*] All operations completed successfully! Enjoy your upscaled picture.")

if __name__ == "__main__":
    main()
