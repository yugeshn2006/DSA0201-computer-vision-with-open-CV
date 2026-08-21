# Experiment 15: Perform Edge detection using Sobel Matrix along XY axis
# Number: 15
# Name: Sobel Edge Detection along XY-Axes

import cv2
import os

def main():
    input_path = os.path.join("images", "landscape_drawing.png")
    output_path = os.path.join("outputs", "Exp15_Sobel_XY.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel X (dx=1, dy=0) and Sobel Y (dx=0, dy=1) separately
    sobelx_64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely_64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Convert absolute values to uint8
    sobelx = cv2.convertScaleAbs(sobelx_64f)
    sobely = cv2.convertScaleAbs(sobely_64f)
    
    # Combine the two gradients (equal weights 0.5 each)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    
    # Save the output image
    cv2.imwrite(output_path, sobelxy)
    print(f"Sobel XY edge detection image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Sobel X", sobelx)
    cv2.imshow("Sobel Y", sobely)
    cv2.imshow("Sobel XY Combined", sobelxy)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
