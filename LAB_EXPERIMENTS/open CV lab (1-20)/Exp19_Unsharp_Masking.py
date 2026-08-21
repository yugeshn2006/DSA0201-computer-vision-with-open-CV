# Experiment 19: Perform Sharpening of Image using unsharp masking.
# Number: 19
# Name: Unsharp Masking

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "house_drawing.png")
    output_path = os.path.join("outputs", "Exp19_Unsharp_Masking.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to float32
    img_float = img.astype(np.float32)

    # Step 1: Blur the original image
    blurred = cv2.GaussianBlur(img_float, (9, 9), 1.5)

    # Step 2: Subtract blurred image from original to get the unsharp mask
    # mask = original - blurred
    mask = img_float - blurred

    # Step 3: Add mask to original to get sharpened image
    # sharpened = original + mask
    sharpened = img_float + mask

    # Clip values to [0, 255] and convert to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    mask_vis = cv2.convertScaleAbs(mask * 2.0) # Scaled for visibility

    # Save output
    cv2.imwrite(output_path, sharpened)
    print(f"Unsharp masked image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Unsharp Mask (Detail Extracted, scaled)", mask_vis)
    cv2.imshow("Sharpened Image (Original + Mask)", sharpened)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
