# Experiment 34: Vehicle Detection in a Video frame using OpenCV.
# Number: 34
# Name: Vehicle Detection using Background Subtraction (MOG2)

import cv2
import numpy as np
import os

def main():
    video_path = os.path.join("images", "vehicle_video.mp4")
    output_video_path = os.path.join("outputs", "Exp34_Vehicle_Detection.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Background subtractor (MOG2) — robust moving object / vehicle detector
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    # Morphological kernel for cleaning the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("Running vehicle detection using MOG2 background subtraction...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        processed = frame.copy()

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Remove shadows (value 127 = shadow in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Denoise: erode then dilate
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:   # Skip tiny noise blobs
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            # Vehicles are generally wider than tall
            if 0.5 < aspect < 5.0 and bh > 20:
                cv2.rectangle(processed, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(processed, f"Vehicle", (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        out.write(processed)

        # Display at half size
        vis = cv2.resize(processed, (width // 2, height // 2))
        cv2.imshow("Vehicle Detection (MOG2)", vis)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vehicle detection video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
