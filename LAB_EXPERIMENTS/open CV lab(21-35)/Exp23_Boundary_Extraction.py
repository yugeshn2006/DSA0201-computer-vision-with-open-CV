# Experiment 23: Find the boundary of the image using Convolution kernel for the given image.
# Number: 23
# Name: Boundary Extraction using Convolution & Morphology

import cv2
import numpy as np
import os

def main():
    # Use mickey_mouse.png as a clear shapes image for morphological boundary extraction
    input_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp23_Boundary_Extraction.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale and threshold to make it a clean binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Method 1: Morphological Boundary Extraction (Binary - Erode(Binary))
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary_morph = cv2.subtract(binary, eroded)

    # Method 2: Custom Convolution Kernel for boundary detection
    # An edge detection convolution kernel (similar to Laplacian)
    # [ -1, -1, -1 ]
    # [ -1,  8, -1 ]
    # [ -1, -1, -1 ]
    conv_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Convolve with the original grayscale image
    boundary_conv_f = cv2.filter2D(gray.astype(np.float32), -1, conv_kernel)
    boundary_conv = cv2.convertScaleAbs(boundary_conv_f)

    # Save the output image
    cv2.imwrite(output_path, boundary_morph)
    print(f"Boundary extraction image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Binary Image", binary)
    cv2.imshow("Boundary via Morphology (I - Erode(I))", boundary_morph)
    cv2.imshow("Boundary via Convolution Kernel", boundary_conv)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
