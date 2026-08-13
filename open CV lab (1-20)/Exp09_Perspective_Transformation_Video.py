# Experiment 9: Perform Perspective Transformation on the Video.
# Number: 9
# Name: Perspective Transformation on Video

import cv2
import numpy as np
import os

def main():
    video_path = os.path.join("images", "vehicle_video.mp4")
    output_video_path = os.path.join("outputs", "Exp09_Perspective_Video.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    # Define VideoWriter to save output (using mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define perspective source points (trapezoid, e.g., simulating road lane perspective)
    # And destination points (rectangle)
    src_pts = np.float32([
        [int(width * 0.15), height],         # Bottom Left
        [int(width * 0.85), height],         # Bottom Right
        [int(width * 0.65), int(height * 0.4)], # Top Right
        [int(width * 0.35), int(height * 0.4)]  # Top Left
    ])
    
    dst_pts = np.float32([
        [0, height],         # Bottom Left
        [width, height],     # Bottom Right
        [width, 0],          # Top Right
        [0, 0]               # Top Left
    ])

    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    print("Processing video perspective transformation and saving output...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Warp the perspective
        warped_frame = cv2.warpPerspective(frame, M, (width, height))
        
        # Write to output video
        out.write(warped_frame)

        # Draw source points trapezoid on the original frame for visualization
        visual_frame = frame.copy()
        pts = src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(visual_frame, [pts], True, (0, 0, 255), 2)
        
        # Display side-by-side or stacked
        # Resize to fit screen if necessary
        vis = np.hstack((cv2.resize(visual_frame, (width // 2, height // 2)), 
                         cv2.resize(warped_frame, (width // 2, height // 2))))
        cv2.imshow("Video Perspective Transformation (Left: Original, Right: Warped)", vis)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Perspective warped video successfully saved to: {output_video_path}")

if __name__ == "__main__":
    main()
