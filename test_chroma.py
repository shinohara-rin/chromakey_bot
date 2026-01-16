import cv2
import numpy as np
import os
import sys
import subprocess

def create_test_image(path):
    # 100x100 green background (0, 255, 0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 255, 0) 
    
    # Red circle in center (0, 0, 255)
    cv2.circle(img, (50, 50), 20, (0, 0, 255), -1)
    
    cv2.imwrite(path, img)
    print(f"Created synthetic image: {path}")

def verify_output(path):
    if not os.path.exists(path):
        print(f"Error: Output file {path} not found")
        return False

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read output image")
        return False
    
    if img.shape[2] != 4:
        print(f"Output has {img.shape[2]} channels, expected 4 (BGRA)")
        return False
        
    # Check corner (0,0) - should be transparent
    corner = img[0, 0]
    if corner[3] != 0:
        print(f"FAILURE: Corner pixel is not transparent. Pixel: {corner}")
        return False
        
    # Check center (50,50) - should be red and opaque
    center = img[50, 50]
    if center[3] != 255:
        print(f"FAILURE: Center pixel is not opaque. Pixel: {center}")
        return False
    
    # Check color preservation (Red: 0, 0, 255)
    if not (center[0] == 0 and center[1] == 0 and center[2] == 255):
        print(f"FAILURE: Center pixel color changed. Expected (0,0,255), got {center[:3]}")
        return False
        
    print("Verification Passed: Background is transparent, subject is opaque.")
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_img = os.path.join(base_dir, "test_input.png")
    output_img = os.path.join(base_dir, "test_output.png")
    script_path = os.path.join(base_dir, "chroma_remove.py")

    create_test_image(input_img)
    
    cmd = [sys.executable, script_path, input_img, output_img]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        if verify_output(output_img):
            print("TEST SUCCEEDED")
            # Cleanup
            if os.path.exists(input_img): os.remove(input_img)
            if os.path.exists(output_img): os.remove(output_img)
        else:
            print("TEST FAILED")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running chroma_remove.py: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
