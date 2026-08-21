# Experiment 28: Morphological operations based on OpenCV using Morphological Gradient technique.
# Number: 28
# Name: Morphological Gradient

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp28_Morphological_Gradient.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply Morphological Gradient
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    # Save output
    cv2.imwrite(output_path, gradient)
    print(f"Morphological gradient image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Binary", binary)
    cv2.imshow("Morphological Gradient (Outlines)", gradient)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
