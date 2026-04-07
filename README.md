# Image Super Resolution Pipeline
**A fully interactive, deep-learning powered Super Resolution CLI built with Python and OpenCV.**

## Overview
This project dynamically upscales input images seamlessly using a diverse suite of classical interpolation techniques (Bilinear, Bicubic, Nearest Neighbor) alongside a modern **Deep Learning AI Super Resolution Model (ESPCN)**. 

The pipeline is completely interactive. It systematically takes your chosen output paths and automatically outputs comprehensive visualizations mapping algorithm performance including Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM).

## Groundbreaking Features
- **3 Operational Modes**:
    1. **Evaluation Mode**: Mathematically downscales an HD reference to simulate data loss, upscales it, and performs strict numerical metric comparisons against the original proxy.
    2. **Practical Mode**: Simply supply any standard image to effortlessly natively scale it massively in high definition without simulated evaluations.
    3. **De-Blockify / Pixel-Art Mode**: Purpose-built pipeline that aggressively downscales and destroys artificially bloated digital compression blocks (heavy JPEG/pixel-art blocks) into their structurally true pixels before feeding them natively to the neural network for clean rendering.
- **Deep Learning ESPCN Integration**: Rapid AI execution via `cv2.dnn_superres` outperforming classical mathematical scaling.
- **Unsharp Post-Processing**: Automated edge clarification routines enhancing blur compensation from raw algorithmic upsizing.
- **Automated Metric Modeling**: Outputs interactive plots scoring mathematical accuracy (`metrics_chart.png`).

## Tech Stack
- **Python**: Core scripting
- **OpenCV & OpenCV-Contrib**: Foundational math array integrations and deep learning SR inferences
- **Scikit-Image**: Evaluation metrics implementations
- **Matplotlib**: Metric chart rendering & Collage plotting

## Setup and Execution
Install dependencies and run interactively:
```bash
# Strongly recommended to run within a virtual python environment
pip install -r Image_Super_Resolution/requirements.txt

# Run the pipeline
python Image_Super_Resolution/src/main.py
```
*Note: Ensure `opencv-contrib-python` is strictly installed over standard `opencv-python` to harness the `cv2.dnn_superres` neural networks properly!*