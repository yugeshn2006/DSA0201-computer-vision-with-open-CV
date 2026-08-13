# Experiment 16: Perform Sharpening of Image using Laplacian mask with negative center coefficient.
# Number: 16
# Name: Laplacian Sharpening with Negative Center Mask

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp16_Laplacian_NegCenter.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to float32 for high precision math
    img_float = img.astype(np.float32)

    # Define the 3x3 Laplacian mask with a negative center coefficient:
    # [ 0,  1,  0]
    # [ 1, -4,  1]
    # [ 0,  1,  0]
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Apply the mask (convolution)
    laplacian = cv2.filter2D(img_float, -1, kernel)

    # For sharpening: subtract the negative-center Laplacian from the original image
    # g(x,y) = f(x,y) - laplacian(x,y)
    sharpened = img_float - laplacian

    # Clip values to [0, 255] and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    laplacian_vis = cv2.convertScaleAbs(laplacian)

    # Save output
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Laplacian Edges (Negative Center)", laplacian_vis)
    cv2.imshow("Sharpened Image (Original - Laplacian)", sharpened)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
