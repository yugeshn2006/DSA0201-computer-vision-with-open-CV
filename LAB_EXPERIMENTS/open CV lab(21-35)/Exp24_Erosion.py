# Experiment 24: Morphological operations based on OpenCV using Erosion technique.
# Number: 24
# Name: Morphological Erosion

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp24_Erosion.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Define kernel (stretching element)
    kernel = np.ones((5, 5), np.uint8)

    # Apply erosion
    eroded = cv2.erode(binary, kernel, iterations=1)

    # Save output
    cv2.imwrite(output_path, eroded)
    print(f"Eroded image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Binary", binary)
    cv2.imshow("Eroded Image", eroded)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
