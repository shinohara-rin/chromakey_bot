import cv2
import numpy as np
import sys
import argparse

def get_corner_samples(image, size=10):
    """
    Extract average color from 4 corners of the image.
    Returns: list of 4 numpy arrays representing average (B, G, R) color.
    """
    h, w = image.shape[:2]
    # Check if image is big enough
    if h < size or w < size:
        raise ValueError("Image is too small for the sampling size")

    # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    corners = [
        image[0:size, 0:size],
        image[0:size, w-size:w],
        image[h-size:h, 0:size],
        image[h-size:h, w-size:w]
    ]
    # Average color for each corner
    means = [np.mean(c, axis=(0, 1)) for c in corners]
    return means

def colors_match(c1, c2, tolerance=10):
    """Check if two colors are within a certain Euclidean distance."""
    return np.linalg.norm(c1 - c2) < tolerance

def remove_chroma(input_path, output_path, tolerance=40):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image {input_path}")

    try:
        means = get_corner_samples(img)
    except Exception as e:
        raise ValueError(f"Error during sampling: {e}")
    
    # Fail fast check: find matching pair to identify background key
    matches = []
    # Indexes: 0:TL, 1:TR, 2:BL, 3:BR
    for i in range(4):
        for j in range(i + 1, 4):
            if colors_match(means[i], means[j], tolerance=20): # Strict tolerance for corner consistency
                matches.append((i, j))
    
    if not matches:
        msg = "Background detection failed. No two corners have a matching color.\nCorner samples (BGR):"
        for idx, m in enumerate(means):
            msg += f"\n  {idx}: {m}"
        raise ValueError(msg)

    # Calculate key color from all matching corners
    idx1, idx2 = matches[0]
    key_color = (means[idx1] + means[idx2]) / 2.0
    print(f"Detected background color (BGR): {key_color}")

    # Compute difference from key color
    diff = np.linalg.norm(img - key_color, axis=2)
    
    # Create mask: True where pixel is close to key color
    mask = diff < tolerance

    # Prepare output with Alpha channel
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=np.uint8) * 255
    else:
        b, g, r, alpha = cv2.split(img)

    # Set alpha to 0 for background pixels
    alpha[mask] = 0

    result = cv2.merge([b, g, r, alpha])
    cv2.imwrite(output_path, result)
    print(f"Success! Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove chromakey background from an image.")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output PNG image")
    parser.add_argument("--tolerance", type=int, default=40, help="Color tolerance for removal (default: 40)")
    
    args = parser.parse_args()
    try:
        remove_chroma(args.input, args.output, args.tolerance)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
