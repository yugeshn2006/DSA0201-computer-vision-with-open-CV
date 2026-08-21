# Experiment 3: Capture video from web Camera and Display the video, in slow motion and in fast motion
# Number: 3
# Name: Live Webcam Playback in Slow and Fast Motion

import cv2
import time
import collections

def main():
    # Attempt to open camera index 0
    print("Attempting to connect to webcam (index 0)...")
    cap = cv2.VideoCapture(0)
    
    # Fallback if webcam is not available
    using_fallback = False
    if not cap.isOpened():
        print("Webcam not found or busy. Falling back to reading from 'images/vehicle_video.mp4' as simulated camera.")
        cap = cv2.VideoCapture("images/vehicle_video.mp4")
        using_fallback = True
        if not cap.isOpened():
            print("Error: Could not open fallback video.")
            return

    print("Press 'n' for Normal speed, 's' for Slow motion, 'f' for Fast motion, and 'q' to Quit.")
    
    mode = "normal"
    frame_buffer = collections.deque(maxlen=150) # Buffer to store frames for slow motion
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if using_fallback:
                # Loop video for continuous simulation
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        frame_count += 1
        
        # Display logic based on mode
        if mode == "normal":
            # Normal real-time display
            cv2.putText(frame, "Mode: NORMAL (Real-time)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Webcam Stream", frame)
            delay = 33
            
        elif mode == "slow":
            # Slow motion: buffer frames and display them with a delay
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > 0:
                display_frame = frame_buffer.popleft()
                cv2.putText(display_frame, "Mode: SLOW MOTION (Buffered)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Webcam Stream", display_frame)
            delay = 100  # Increase delay to slow down display
            
        elif mode == "fast":
            # Fast motion: skip frames (only show every 3rd frame)
            if frame_count % 3 == 0:
                cv2.putText(frame, "Mode: FAST MOTION (Skipping 2/3 frames)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow("Webcam Stream", frame)
            delay = 11  # Short delay for fast playback
            
        # Check for key presses
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("Quitting webcam stream.")
            break
        elif key == ord('n'):
            mode = "normal"
            frame_buffer.clear()
            print("Switched to Normal mode.")
        elif key == ord('s'):
            mode = "slow"
            frame_buffer.clear()
            print("Switched to Slow Motion mode.")
        elif key == ord('f'):
            mode = "fast"
            frame_buffer.clear()
            print("Switched to Fast Motion mode.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
