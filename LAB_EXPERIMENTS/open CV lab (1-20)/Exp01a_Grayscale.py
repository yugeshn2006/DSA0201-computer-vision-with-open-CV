# Experiment 1a: Read an image in python and Convert the given Image into Grayscale
# Number: 1a
# Name: Convert Image to Grayscale

import cv2
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp01a_Grayscale.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save the output image
    cv2.imwrite(output_path, gray_img)
    print(f"Grayscale image saved successfully to: {output_path}")
    
    # Display the result
    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", gray_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
