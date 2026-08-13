# Experiment 1c: Read an image in python and Convert the given Image to show outline using Canny function.
# Number: 1c
# Name: Convert Image to Show Outline using Canny

import cv2
import os

def main():
    input_path = os.path.join("images", "landscape_drawing.png")
    output_path = os.path.join("outputs", "Exp01c_CannyOutline.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to Grayscale first, as Canny works on single-channel images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Perform Canny Edge Detection
    # Thresholds are set to 100 and 200
    outline_img = cv2.Canny(gray, 100, 200)
    
    # Save the output image
    cv2.imwrite(output_path, outline_img)
    print(f"Canny outline image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Image", img)
    cv2.imshow("Canny Outline", outline_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
