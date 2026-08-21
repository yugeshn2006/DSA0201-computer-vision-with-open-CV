# Experiment 18: Perform Sharpening of Image using Laplacian mask with positive center coefficient.
# Number: 18
# Name: Laplacian Sharpening with Positive Center Mask (Addition)

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp18_Laplacian_PosCenter.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to float32
    img_float = img.astype(np.float32)

    # Define the 3x3 positive center sharpening mask:
    # [ 0, -1,  0]
    # [-1,  5, -1]
    # [ 0, -1,  0]
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Perform direct sharpening convolution
    sharpened = cv2.filter2D(img_float, -1, kernel)

    # Clip to [0, 255] and convert to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Save output
    cv2.imwrite(output_path, sharpened)
    print(f"Sharpened image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Sharpened Image (Positive Center Mask)", sharpened)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
