# Experiment 4: Scaling an image to its Bigger and Smaller sizes.
# Number: 4
# Name: Image Scaling (Bigger and Smaller)

import cv2
import os

def main():
    input_path = os.path.join("images", "pink_tree.png")
    output_bigger_path = os.path.join("outputs", "Exp04_Scaling_Bigger.png")
    output_smaller_path = os.path.join("outputs", "Exp04_Scaling_Smaller.png")
    
    # Read the image
    print(f"Reading image from: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}. Please run create_test_assets.py first.")
        return

    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")

    # Scale to a bigger size (2x larger)
    # Using INTER_CUBIC or INTER_LINEAR for high-quality enlargement
    bigger_img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    print(f"Bigger image size: {w*2}x{h*2}")

    # Scale to a smaller size (0.5x smaller)
    # Using INTER_AREA for high-quality downsampling (prevents aliasing)
    smaller_img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    print(f"Smaller image size: {w//2}x{h//2}")

    # Save output images
    cv2.imwrite(output_bigger_path, bigger_img)
    cv2.imwrite(output_smaller_path, smaller_img)
    print("Resized images saved successfully.")
    
    # Display the results
    # We display original and smaller images directly, 
    # and the bigger image in a resized window to fit screens if necessary
    cv2.imshow("Original Image", img)
    cv2.imshow("Smaller Image (0.5x)", smaller_img)
    cv2.imshow("Bigger Image (2x)", bigger_img)
    
    print("Displaying windows. Click on any window and press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
