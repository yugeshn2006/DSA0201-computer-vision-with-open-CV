# Experiment 1b: Read an image in python and Convert an Image to Blur using GaussianBlur
# Number: 1b
# Name: Convert Image to Blur using GaussianBlur

import cv2
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp01b_GaussianBlur.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to Blur using GaussianBlur (using kernel size 15x15)
    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Save the output image
    cv2.imwrite(output_path, blurred_img)
    print(f"Blurred image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Image", img)
    cv2.imshow("Gaussian Blurred Image", blurred_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
