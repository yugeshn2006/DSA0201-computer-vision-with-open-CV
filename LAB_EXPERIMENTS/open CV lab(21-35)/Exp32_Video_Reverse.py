# Experiment 32: Using Opencv play Video in Reverse mode.
# Number: 32
# Name: Play Video in Reverse Mode

import cv2
import os

def main():
    video_path = os.path.join("images", "vehicle_video.mp4")
    output_video_path = os.path.join("outputs", "Exp32_Video_Reverse.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read all frames into a list
    print("Reading video frames into memory...")
    frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Total frames read: {len(frames)}")

    if len(frames) == 0:
        print("Error: No frames found in video.")
        return

    # Set up VideoWriter to save reversed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Playing and saving video in reverse...")
    # Iterate in reverse order
    for i in reversed(range(len(frames))):
        frame = frames[i]
        
        # Add overlay indicating reverse play
        visual_frame = frame.copy()
        cv2.putText(visual_frame, "REVERSE PLAYBACK", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Reverse Video Playback", visual_frame)
        out.write(frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Reversed video successfully saved to: {output_video_path}")

if __name__ == "__main__":
    main()
