# Experiment 26: Morphological operations based on OpenCV using Opening technique.
# Number: 26
# Name: Morphological Opening

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp26_Opening.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Let's add some artificial white noise dots in the background to show how Opening removes them
    h, w = binary.shape
    noisy_binary = binary.copy()
    # Add random white noise pixels
    for _ in range(500):
        ny = np.random.randint(0, h)
        nx = np.random.randint(0, w)
        noisy_binary[ny, nx] = 255

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply Morphological Opening
    opened = cv2.morphologyEx(noisy_binary, cv2.MORPH_OPEN, kernel)

    # Save output
    cv2.imwrite(output_path, opened)
    print(f"Opened image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Noisy Input Binary", noisy_binary)
    cv2.imshow("After Morphological Opening (Noise Removed)", opened)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
