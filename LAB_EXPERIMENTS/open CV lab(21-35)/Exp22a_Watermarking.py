# Experiment 22a: Insert water marking to the image using OpenCV.
# Number: 22a
# Name: Image Watermarking (Text and Logo)

import cv2
import numpy as np
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    logo_path = os.path.join("images", "mickey_mouse.png")
    output_path = os.path.join("outputs", "Exp22a_Watermarking.png")
    
    # Read primary image
    print(f"Reading main image from: {input_path}")
    img = cv2.imread(input_path)
    
    # Read logo image
    print(f"Reading logo image from: {logo_path}")
    logo = cv2.imread(logo_path)
    
    if img is None or logo is None:
        print("Error: Could not read image or logo. Please run create_test_assets.py first.")
        return

    # Method 1: Text Watermark with transparency
    # We write onto an overlay layer and blend it
    text_overlay = img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    text = "CONFIDENTIAL - LAB EXP 22A"
    
    # Calculate text size and center it
    h, w = img.shape[:2]
    text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
    tx = (w - text_size[0]) // 2
    ty = (h + text_size[1]) // 2
    
    # Draw dark gray text on the overlay
    cv2.putText(text_overlay, text, (tx, ty), font, 1.5, (100, 100, 100), 3, cv2.LINE_AA)
    
    # Blend the text overlay with original (opacity 0.6)
    watermarked = cv2.addWeighted(text_overlay, 0.4, img, 0.6, 0)

    # Method 2: Image Logo Watermark in the bottom-right corner
    # Resize logo to fit corner (say, 100x100 pixels)
    logo_h, logo_w = 120, 120
    logo_resized = cv2.resize(logo, (logo_w, logo_h))
    
    # Region of Interest (ROI) on watermarked image where logo will be placed
    margin = 20
    roi_x = w - logo_w - margin
    roi_y = h - logo_h - margin
    roi = watermarked[roi_y:roi_y+logo_h, roi_x:roi_x+logo_w]
    
    # Overlay the logo using blending
    # (Since mickey_mouse has a white background, simple blending works well)
    logo_blend = cv2.addWeighted(roi, 0.5, logo_resized, 0.5, 0)
    watermarked[roi_y:roi_y+logo_h, roi_x:roi_x+logo_w] = logo_blend

    # Save output
    cv2.imwrite(output_path, watermarked)
    print(f"Watermarked image saved successfully to: {output_path}")

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Watermarked Image (Text & Logo)", watermarked)
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
