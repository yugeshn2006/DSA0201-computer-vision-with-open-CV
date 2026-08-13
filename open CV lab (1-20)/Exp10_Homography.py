# Experiment 10: Perform transformation using Homography matrix.
# Number: 10
# Name: Image Transformation using Homography Matrix

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp10_Homography.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # Source points (coordinates of the original image corners)
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Destination points (where we want the corners to map in the output)
    # Warping the image into a stylized perspective quadrant
    dst_pts = np.array([
        [int(w * 0.1), int(h * 0.2)],
        [int(w * 0.85), int(h * 0.05)],
        [int(w * 0.95), int(h * 0.9)],
        [int(w * 0.05), int(h * 0.75)]
    ], dtype=np.float32)

    # Find the Homography matrix
    # cv2.findHomography returns the homography matrix and a mask (for RANSAC)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    print("\nCalculated Homography Matrix (3x3):")
    print(H)

    # Warp the image using the homography matrix
    homography_img = cv2.warpPerspective(img, H, (w, h))

    # Save output
    cv2.imwrite(output_path, homography_img)
    print(f"\nHomography transformed image saved successfully to: {output_path}")

    # Display the results
    cv2.imshow("Original Image", img)
    cv2.imshow("Homography Transformed", homography_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
