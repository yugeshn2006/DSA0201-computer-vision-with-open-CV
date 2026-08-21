# Experiment 22b: Do Cropping, Copying and pasting image inside another image using OpenCV.
# Number: 22b
# Name: Image Cropping, Copying, and Pasting ROI

import cv2
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp22b_Cropping_Pasting.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # Step 1: Crop a Region of Interest (ROI)
    # Let's crop a central portion (e.g., the pink tree foliage)
    # Coordinates: y from 0.2*h to 0.7*h, x from 0.25*w to 0.75*w
    y1, y2 = int(h * 0.2), int(h * 0.7)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    
    print(f"Cropping ROI: y[{y1}:{y2}], x[{x1}:{x2}]")
    roi = img[y1:y2, x1:x2]

    # Step 2: Copy the cropped ROI
    roi_copy = roi.copy()

    # Step 3: Paste it onto another region of the image
    # Let's paste a scaled-down version of the ROI into the top-left corner
    roi_h, roi_w = roi.shape[:2]
    new_h, new_w = int(roi_h * 0.4), int(roi_w * 0.4)
    roi_resized = cv2.resize(roi_copy, (new_w, new_h))
    
    # Target coordinates: top-left corner with 20px padding
    ty1, ty2 = 20, 20 + new_h
    tx1, tx2 = 20, 20 + new_w
    
    # Paste ROI copy
    result_img = img.copy()
    result_img[ty1:ty2, tx1:tx2] = roi_resized
    
    # Draw boxes to show where we cropped from and pasted to
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red = Source crop area
    cv2.rectangle(result_img, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2) # Green = Target paste area
    
    # Save the output image
    cv2.imwrite(output_path, result_img)
    print(f"Result image saved successfully to: {output_path}")

    # Display the results
    cv2.imshow("Original Image", img)
    cv2.imshow("Cropped ROI", roi)
    cv2.imshow("Image after Copy-Paste (Red=Source, Green=Dest)", result_img)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
