# Experiment 2: Read captured video in python and display the video, in slow motion and in fast motion
# Number: 2
# Name: Play Video in Slow Motion and Fast Motion

import cv2
import os

def play_video(video_path, mode="normal"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Determine delay based on mode
    # Normal: ~30 FPS -> 33ms delay
    # Slow motion: slower playback -> 100ms delay
    # Fast motion: faster playback -> 10ms delay (or skip frames)
    if mode == "slow":
        delay = 100
        print("Playing in SLOW MOTION (Delay: 100ms)...")
    elif mode == "fast":
        delay = 10
        print("Playing in FAST MOTION (Delay: 10ms)...")
    else:
        delay = 33
        print("Playing at NORMAL speed (Delay: 33ms)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, f"Mode: {mode.upper()}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Playback", frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = os.path.join("images", "vehicle_video.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found. Please run create_test_assets.py first.")
        return

    # Sequential playback of normal, slow, and fast speeds
    play_video(video_path, mode="normal")
    play_video(video_path, mode="slow")
    play_video(video_path, mode="fast")

if __name__ == "__main__":
    main()
