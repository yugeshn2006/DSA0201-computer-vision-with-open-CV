# Experiment 27: Morphological operations based on OpenCV using Closing technique.
# Number: 27
# Name: Morphological Closing

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp27_Closing.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Let's add some artificial black holes (pepper noise) inside the mouse shape to show how Closing fills them
    h, w = binary.shape
    noisy_binary = binary.copy()
    # Add random black noise pixels in the center area where shape exists
    for _ in range(500):
        ny = np.random.randint(int(h * 0.3), int(h * 0.7))
        nx = np.random.randint(int(w * 0.3), int(w * 0.7))
        if noisy_binary[ny, nx] == 255:
            noisy_binary[ny, nx] = 0

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply Morphological Closing
    closed = cv2.morphologyEx(noisy_binary, cv2.MORPH_CLOSE, kernel)

    # Save output
    cv2.imwrite(output_path, closed)
    print(f"Closed image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Noisy Input Binary", noisy_binary)
    cv2.imshow("After Morphological Closing (Holes Filled)", closed)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
