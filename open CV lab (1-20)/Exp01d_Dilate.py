# Experiment 1d: Read an image in python and Dilate an Image using Dilate function.
# Number: 1d
# Name: Dilate an Image using Dilate

import cv2
import numpy as np
import os

def main():
    # Using mickey_mouse.png for morphological operations as it has high-contrast structures
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp01d_Dilate.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Define a 5x5 structuring element (kernel) of ones
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform Dilation
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    
    # Save the output image
    cv2.imwrite(output_path, dilated_img)
    print(f"Dilated image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Image", img)
    cv2.imshow("Dilated Image", dilated_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
