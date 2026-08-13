# Experiment 13: Perform Edge detection using Sobel Matrix along X axis
# Number: 13
# Name: Sobel Edge Detection along X-Axis

import cv2
import os

def main():
    input_path = os.path.join("images", "landscape_drawing.png")
    output_path = os.path.join("outputs", "Exp13_Sobel_X.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel along X axis (dx=1, dy=0)
    # Using 64-bit float representation to handle negative gradients, then converting back
    sobelx_64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx_64f)
    
    # Save the output image
    cv2.imwrite(output_path, sobelx)
    print(f"Sobel X edge detection image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Sobel X (Horizontal Gradients)", sobelx)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
