# Experiment 11: Perform transformation using Direct Linear Transformation.
# Number: 11
# Name: Direct Linear Transformation (DLT) from Scratch

import cv2
import numpy as np
import os

def estimate_homography_dlt(src_pts, dst_pts):
    """
    Estimates the 3x3 Homography matrix using Direct Linear Transformation (DLT).
    src_pts and dst_pts are arrays of shape (N, 2), containing corresponding points.
    N must be >= 4.
    """
    num_pts = src_pts.shape[0]
    A = []
    
    for i in range(num_pts):
        x, y = src_pts[i][0], src_pts[i][1]
        xp, yp = dst_pts[i][0], dst_pts[i][1]
        
        # Row 1: [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        # Row 2: [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        
    A = np.array(A)
    
    # Solve Ah = 0 using SVD
    # A = U * S * V^T
    # The solution h is the last row of V^T (or last column of V)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    
    # Reshape h into a 3x3 matrix
    H = h.reshape((3, 3))
    
    # Normalize H so that H[2,2] is 1.0
    if H[2, 2] != 0:
        H = H / H[2, 2]
        
    return H

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp11_DLT.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # Source points (four corners of the original image)
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Destination points (corresponds to a perspective slant)
    dst_pts = np.array([
        [int(w * 0.15), int(h * 0.15)],
        [int(w * 0.8), int(h * 0.05)],
        [int(w * 0.9), int(h * 0.85)],
        [int(w * 0.05), int(h * 0.95)]
    ], dtype=np.float32)

    # Compute Homography using DLT from scratch
    print("Estimating homography matrix using DLT...")
    H_dlt = estimate_homography_dlt(src_pts, dst_pts)
    
    # Compare with OpenCV findHomography for validation
    H_cv2, _ = cv2.findHomography(src_pts, dst_pts)
    
    print("\n--- Homography Comparison ---")
    print("DLT Homography (scratch):")
    print(H_dlt)
    print("\nOpenCV findHomography:")
    print(H_cv2)
    print("\nAbsolute Difference:")
    print(np.abs(H_dlt - H_cv2))

    # Warp image using computed homography
    warped_img = cv2.warpPerspective(img, H_dlt, (w, h))

    # Save output
    cv2.imwrite(output_path, warped_img)
    print(f"\nDLT transformed image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("DLT Perspective Warp", warped_img)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
