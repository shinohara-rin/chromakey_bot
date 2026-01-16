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

import os

def process_image(img, tolerance=40):
    """
    Process image to remove chromakey background.
    Args:
        img: numpy array (BGR)
        tolerance: color tolerance
    Returns:
        numpy array (BGRA) with transparent background
    """
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

    # Auto-crop logic
    # Find all non-zero alpha pixels
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Crop to bounding box
        b = b[y:y+h, x:x+w]
        g = g[y:y+h, x:x+w]
        r = r[y:y+h, x:x+w]
        alpha = alpha[y:y+h, x:x+w]

    return cv2.merge([b, g, r, alpha])

def remove_chroma(input_path, output_path, tolerance=40):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return

    try:
        result = process_image(img, tolerance)
        cv2.imwrite(output_path, result)
        print(f"Success! Saved result to {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove chromakey background from an image.")
    parser.add_argument("inputs", nargs="+", help="Path to input image(s)")
    parser.add_argument("-o", "--output", help="Path to output PNG image or directory")
    parser.add_argument("-i", "--inplace", action="store_true", help="Edit inplace (save as PNG in same directory)")
    parser.add_argument("--tolerance", type=int, default=40, help="Color tolerance for removal (default: 40)")
    
    args = parser.parse_args()

    if args.inplace and args.output:
        print("Error: Cannot specify both --output and --inplace")
        sys.exit(1)

    if not args.inplace and not args.output:
        print("Error: Must specify either --output or --inplace")
        sys.exit(1)

    # Check if multiple inputs but output is a file (and not inplace)
    if len(args.inputs) > 1 and not args.inplace:
        if args.output and not os.path.isdir(args.output):
            # If it doesn't exist, we can create it if it looks like a directory? 
            # Or better, just require it to be an existing dir or end with slash? 
            # For simplicity: if it has an extension, it's likely a file.
            # But safer: if multiple inputs, output MUST be a dir. 
            # We will create it if it doesn't exist.
            try:
                os.makedirs(args.output, exist_ok=True)
            except Exception:
                print(f"Error: When processing multiple files, output must be a directory. Could not create {args.output}")
                sys.exit(1)
    
    for input_path in args.inputs:
        input_filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(input_filename)
        # Always output png
        output_filename = f"{base_name}.png"

        if args.inplace:
            # Save to same directory as input
            output_dir = os.path.dirname(input_path)
            final_output_path = os.path.join(output_dir, output_filename)
        else:
            if os.path.isdir(args.output):
                # Save to output directory
                final_output_path = os.path.join(args.output, output_filename)
            else:
                # Output is a file path
                final_output_path = args.output

        remove_chroma(input_path, final_output_path, args.tolerance)
