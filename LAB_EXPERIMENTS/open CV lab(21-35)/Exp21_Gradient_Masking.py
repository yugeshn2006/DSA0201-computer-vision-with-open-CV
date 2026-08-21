# Experiment 21: Perform Sharpening of Image using Gradient masking.
# Number: 21
# Name: Gradient Sharpening Mask

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp21_Gradient_Masking.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to float32
    img_float = img.astype(np.float32)

    # Compute gradients along X and Y axes using Sobel operator
    # Convert to grayscale first for gradient computation, but we can apply it to color channels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude: G = sqrt(Gx^2 + Gy^2)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize gradient magnitude to [0, 255] for visualization
    gradient_vis = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Add gradient back to the original image to sharpen
    # We do this for each of the B, G, R channels
    sharpened = np.zeros_like(img_float)
    # Scale factor for gradient addition
    k = 0.5
    for i in range(3):
        sharpened[:, :, i] = img_float[:, :, i] + k * gradient_magnitude

    # Clip to [0, 255] and convert to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Save output
    cv2.imwrite(output_path, sharpened)
    print(f"Gradient-sharpened image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Gradient Magnitude (Edges)", gradient_vis)
    cv2.imshow("Gradient Sharpened (Original + k*Gradient)", sharpened)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
