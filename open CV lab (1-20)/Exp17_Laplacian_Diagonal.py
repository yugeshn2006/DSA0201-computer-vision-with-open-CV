# Experiment 17: Perform Sharpening of Image using Laplacian mask implemented with an extension of diagonal neighbors.
# Number: 17
# Name: Laplacian Sharpening with Diagonal Neighbors Mask

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp17_Laplacian_Diagonal.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to float32
    img_float = img.astype(np.float32)

    # Define the 3x3 diagonal extension Laplacian kernel:
    # [ 1,  1,  1]
    # [ 1, -8,  1]
    # [ 1,  1,  1]
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=np.float32)

    # Convolve the image with the kernel
    laplacian = cv2.filter2D(img_float, -1, kernel)

    # Sharpen: subtract the Laplacian (since it has a negative center)
    sharpened = img_float - laplacian

    # Clip to [0, 255] and convert to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    laplacian_vis = cv2.convertScaleAbs(laplacian)

    # Save output
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image (diagonal Laplacian) saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Diagonal Laplacian Edges", laplacian_vis)
    cv2.imshow("Sharpened Image (Diagonal)", sharpened)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
