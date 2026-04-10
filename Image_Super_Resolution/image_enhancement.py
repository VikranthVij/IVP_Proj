import cv2
import numpy as np


def denoise_before_sharpen(image):
    return cv2.bilateralFilter(image, 5, 75, 75)


def advanced_sharpen(image, strength=1.8, edge_boost=1.5):
    """
    Multi-stage sharpening:
    1. Unsharp masking
    2. Edge enhancement
    """

    img = image.astype(np.float32)

    # Blur (for unsharp masking)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)

    # Unsharp masking
    sharp = img + strength * (img - blurred)

    # Laplacian edge detection
    edges = cv2.Laplacian(img, cv2.CV_32F)

    # Edge boost
    enhanced = sharp + edge_boost * edges

    # Clip values
    enhanced = np.clip(enhanced, 0, 255)

    return enhanced.astype(np.uint8)