# Experiment 8: Perform Perspective Transformation on the image.
# Number: 8
# Name: Perspective Transformation on Image

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp08_Perspective.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # Define four corner points in the source image
    # For demonstration, we warp the top half inward to create a perspective slant
    src_pts = np.float32([
        [0, 0],          # Top Left
        [w - 1, 0],      # Top Right
        [0, h - 1],      # Bottom Left
        [w - 1, h - 1]   # Bottom Right
    ])

    # Define where they should map in the output image
    dst_pts = np.float32([
        [int(w * 0.25), int(h * 0.1)],  # Top Left moved inward
        [int(w * 0.75), int(h * 0.1)],  # Top Right moved inward
        [0, h - 1],                     # Bottom Left remains same
        [w - 1, h - 1]                  # Bottom Right remains same
    ])

    # Get perspective transform matrix M
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp perspective
    warped_img = cv2.warpPerspective(img, M, (w, h))

    # Save output
    cv2.imwrite(output_path, warped_img)
    print(f"Perspective transformed image saved to: {output_path}")

    # Draw reference points on source for visualization
    visual_img = img.copy()
    for pt in src_pts:
        cv2.circle(visual_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)

    # Display the result
    cv2.imshow("Original with Source Corners", visual_img)
    cv2.imshow("Perspective Warped Image", warped_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
