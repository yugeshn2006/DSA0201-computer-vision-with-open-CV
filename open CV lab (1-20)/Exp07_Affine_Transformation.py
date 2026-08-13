# Experiment 7: Perform Affine Transformation on the image.
# Number: 7
# Name: Affine Transformation

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp07_Affine.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # Define three reference points on the original image
    pts1 = np.float32([
        [50, 50],
        [w - 100, 50],
        [50, h - 100]
    ])

    # Define their new locations in the output image
    pts2 = np.float32([
        [10, 100],
        [w - 50, 50],
        [100, h - 50]
    ])

    # Get the 2x3 affine transformation matrix
    M = cv2.getAffineTransform(pts1, pts2)

    # Warp the image
    affine_img = cv2.warpAffine(img, M, (w, h))

    # Save the output image
    cv2.imwrite(output_path, affine_img)
    print(f"Affine transformed image saved successfully to: {output_path}")
    
    # Draw reference points on original and transformed image for visualization
    visual_img = img.copy()
    visual_affine = affine_img.copy()
    for p1, p2 in zip(pts1, pts2):
        cv2.circle(visual_img, tuple(p1.astype(int)), 8, (0, 0, 255), -1)
        cv2.circle(visual_affine, tuple(p2.astype(int)), 8, (0, 255, 0), -1)

    # Display the result
    cv2.imshow("Original Image with Reference Points", visual_img)
    cv2.imshow("Affine Transformed Image", visual_affine)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
