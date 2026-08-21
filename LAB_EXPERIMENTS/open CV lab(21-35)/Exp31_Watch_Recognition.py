# Experiment 31: Recognise watch from the given image by general Object recognition using OpenCV.
# Number: 31
# Name: General Object Recognition (Watch Recognition)

import cv2
import numpy as np
import os

def main():
    image_path = os.path.join("images", "watch.png")
    template_path = os.path.join("images", "watch_template.png")
    output_path = os.path.join("outputs", "Exp31_Watch_Recognition.png")
    
    # Read the main search image and the template
    print(f"Reading search image from: {image_path}")
    img = cv2.imread(image_path)
    print(f"Reading template image from: {template_path}")
    template = cv2.imread(template_path)
    
    if img is None or template is None:
        print("Error: Could not read watch image or template. Please run create_test_assets.py first.")
        return

    # Convert both to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    w, h = gray_template.shape[1], gray_template.shape[0]

    # Perform Template Matching
    # Using TM_CCOEFF_NORMED which handles lighting variations relatively well
    print("Performing template matching for watch recognition...")
    res = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Find min and max value locations in template match score matrix
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # For TM_CCOEFF_NORMED, the best match location is max_loc
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    print(f"Watch recognized! Confidence: {max_val * 100:.2f}%")
    print(f"Bounding Box: Top-Left={top_left}, Bottom-Right={bottom_right}")

    # Draw a bounding box around the detected template area
    result_img = img.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(result_img, f"WATCH ({max_val * 100:.1f}%)", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Save output
    cv2.imwrite(output_path, result_img)
    print(f"Object recognition result saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Template (Watch Face)", template)
    cv2.imshow("Recognized Watch Object", result_img)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
