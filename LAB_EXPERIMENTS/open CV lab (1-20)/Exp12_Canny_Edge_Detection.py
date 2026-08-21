# Experiment 12: Perform Edge detection using canny method.
# Number: 12
# Name: Interactive Canny Edge Detection

import cv2
import os

def nothing(x):
    pass

def main():
    input_path = os.path.join("images", "landscape_drawing.png")
    output_path = os.path.join("outputs", "Exp12_Canny_Edge_Detection.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply soft Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create window with trackbars for interactive tuning
    cv2.namedWindow("Canny Edge Detector")
    cv2.createTrackbar("Low Threshold", "Canny Edge Detector", 50, 255, nothing)
    cv2.createTrackbar("High Threshold", "Canny Edge Detector", 150, 255, nothing)

    print("Dynamic Canny edge detection. Adjust the trackbars in the window.")
    print("Press 's' to save the current edge map and exit. Press 'q' to quit without saving.")

    while True:
        # Get current trackbar positions
        low = cv2.getTrackbarPos("Low Threshold", "Canny Edge Detector")
        high = cv2.getTrackbarPos("High Threshold", "Canny Edge Detector")
        
        # Apply Canny
        edges = cv2.Canny(blurred, low, high)
        
        # Display results
        cv2.imshow("Canny Edge Detector", edges)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            cv2.imwrite(output_path, edges)
            print(f"Selected edge map saved successfully to: {output_path}")
            break
        elif key == ord('q') or key == 27: # Esc or 'q'
            # Save default state anyway so output is present
            cv2.imwrite(output_path, edges)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
