# Experiment 35: Draw Rectangular shape and extract objects.
# Number: 35
# Name: Draw Rectangular Shape and Extract Object

import cv2
import os

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = -1, -1, -1, -1
img_clean = None
img_display = None

def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, rx, ry, rw, rh, img_display, img_clean
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img_clean.copy()
            # Draw temporary rectangle while dragging
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_display = img_clean.copy()
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
        
        # Calculate bounding box coordinates
        rx = min(ix, x)
        ry = min(iy, y)
        rw = abs(ix - x)
        rh = abs(iy - y)

def main():
    global img_clean, img_display, rx, ry, rw, rh
    
    input_path = os.path.join("images", "pink_tree.png")
    output_path = os.path.join("outputs", "Exp35_ExtractedObject.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img_clean = cv2.imread(input_path)
    
    if img_clean is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    img_display = img_clean.copy()

    # Create window and bind mouse callback
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", draw_rectangle)

    print("\n--- INSTRUCTIONS ---")
    print("1. Click and drag the mouse on the window to draw a green rectangle around the tree or any object.")
    print("2. Release the mouse button to finalize.")
    print("3. Press 'e' to EXTRACT and save the selected object.")
    print("4. Press 'q' to quit.")

    while True:
        cv2.imshow("Select Region", img_display)
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('e'):
            if rw > 5 and rh > 5:
                # Extract the cropped region
                cropped = img_clean[ry:ry+rh, rx:rx+rw]
                cv2.imshow("Extracted Object", cropped)
                cv2.imwrite(output_path, cropped)
                print(f"Extracted object successfully saved to: {output_path}")
            else:
                print("Warning: Draw a valid rectangle first before pressing 'e'.")
                
        elif key == ord('q') or key == 27: # 'q' or Esc
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
