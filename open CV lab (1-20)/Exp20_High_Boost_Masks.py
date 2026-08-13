# Experiment 20: Perform Sharpening of Image using High-Boost Masks.
# Number: 20
# Name: High-Boost Filtering

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp20_High_Boost.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to float32
    img_float = img.astype(np.float32)

    # Blur the original image
    blurred = cv2.GaussianBlur(img_float, (9, 9), 1.5)

    # High-Boost Filtering formula:
    # g(x, y) = A * f(x, y) - blurred(x, y)
    # Let's use boost factor A = 1.5
    A = 1.5
    print(f"Applying High-Boost Filtering with A = {A}...")
    
    high_boost = A * img_float - blurred

    # Clip to [0, 255] and convert to uint8
    high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)

    # Save output
    cv2.imwrite(output_path, high_boost)
    print(f"High-boost sharpened image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("High-Boost Sharpened (A=1.5)", high_boost)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
