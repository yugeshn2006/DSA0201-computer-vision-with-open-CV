# Experiment 25: Morphological operations based on OpenCV using Dilation technique.
# Number: 25
# Name: Morphological Dilation

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp25_Dilation.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Define kernel (structuring element)
    kernel = np.ones((5, 5), np.uint8)

    # Apply dilation
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Save output
    cv2.imwrite(output_path, dilated)
    print(f"Dilated image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Binary", binary)
    cv2.imshow("Dilated Image", dilated)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
