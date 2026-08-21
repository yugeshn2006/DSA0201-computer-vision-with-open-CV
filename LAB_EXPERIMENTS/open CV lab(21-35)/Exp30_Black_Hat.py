# Experiment 30: Morphological operations based on OpenCV using Black hat technique.
# Number: 30
# Name: Morphological Black Hat Transform

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp30_Black_Hat.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define a large structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # Apply Morphological Black Hat
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Save output
    cv2.imwrite(output_path, blackhat)
    print(f"Black hat transform image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Black Hat Transform (Dark details isolated)", blackhat)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
