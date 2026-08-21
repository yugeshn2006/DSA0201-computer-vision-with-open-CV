# Experiment 5: Perform Rotation of an image to clockwise and counter clockwise direction.
# Number: 5
# Name: Image Rotation (Clockwise and Counter-Clockwise)

import cv2
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_cw_path = os.path.join("outputs", "Exp05_Rotation_CW.png")
    output_ccw_path = os.path.join("outputs", "Exp05_Rotation_CCW.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Method 1: Exact 90-degree rotations
    # Clockwise 90 degrees
    img_cw_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # Counter-clockwise 90 degrees
    img_ccw_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Method 2: Arbitrary angle rotation (e.g., 45 degrees) using Rotation Matrix
    # Get image center
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate 45 degrees clockwise rotation matrix (negative angle in OpenCV is clockwise, positive is counter-clockwise)
    matrix_cw = cv2.getRotationMatrix2D(center, -45, 1.0)
    img_cw_45 = cv2.warpAffine(img, matrix_cw, (w, h))

    # Calculate 45 degrees counter-clockwise rotation matrix
    matrix_ccw = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_ccw_45 = cv2.warpAffine(img, matrix_ccw, (w, h))

    # Save outputs (we'll save the 90 degree ones as standard)
    cv2.imwrite(output_cw_path, img_cw_90)
    cv2.imwrite(output_ccw_path, img_ccw_90)
    print("Rotated images saved successfully.")
    
    # Display the results
    cv2.imshow("Original Image", img)
    cv2.imshow("90 Deg CW", img_cw_90)
    cv2.imshow("90 Deg CCW", img_ccw_90)
    cv2.imshow("45 Deg CW (Warped)", img_cw_45)
    cv2.imshow("45 Deg CCW (Warped)", img_ccw_45)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
