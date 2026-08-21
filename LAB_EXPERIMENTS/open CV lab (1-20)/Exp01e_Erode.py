# Experiment 1e: Read an image in python and Erode an Image using erode function.
# Number: 1e
# Name: Erode an Image using Erode

import cv2
import numpy as np
import os

def main():
    # Using purple_flower.png as indicated in the PDF screenshots
    input_path = os.path.join("images", "purple_flower.png")
    output_path = os.path.join("outputs", "Exp01e_Erode.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Define a 5x5 structuring element (kernel) of ones
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform Erosion
    eroded_img = cv2.erode(img, kernel, iterations=1)
    
    # Save the output image
    cv2.imwrite(output_path, eroded_img)
    print(f"Eroded image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Image", img)
    cv2.imshow("Eroded Image", eroded_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
