# Experiment 6: Perform moving of an image from one place to another.
# Number: 6
# Name: Image Translation (Shift Location)

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp06_Translation.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # Define shifts: tx = 100 pixels (right), ty = 50 pixels (down)
    tx, ty = 100, 50
    print(f"Translating image by tx={tx} pixels, ty={ty} pixels...")

    # Create translation matrix
    # T = [[1, 0, tx],
    #      [0, 1, ty]]
    T = np.float32([[1, 0, tx], [0, 1, ty]])

    # Apply warpAffine to translate
    translated_img = cv2.warpAffine(img, T, (w, h))

    # Save the output image
    cv2.imwrite(output_path, translated_img)
    print(f"Translated image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Image", img)
    cv2.imshow("Translated Image (Shifted Right/Down)", translated_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
