# Experiment 14: Perform Edge detection using Sobel Matrix along Y axis
# Number: 14
# Name: Sobel Edge Detection along Y-Axis

import cv2
import os

def main():
    input_path = os.path.join("images", "landscape_drawing.png")
    output_path = os.path.join("outputs", "Exp14_Sobel_Y.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel along Y axis (dx=0, dy=1)
    sobely_64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely_64f)
    
    # Save the output image
    cv2.imwrite(output_path, sobely)
    print(f"Sobel Y edge detection image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Grayscale", gray)
    cv2.imshow("Sobel Y (Vertical Gradients)", sobely)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
