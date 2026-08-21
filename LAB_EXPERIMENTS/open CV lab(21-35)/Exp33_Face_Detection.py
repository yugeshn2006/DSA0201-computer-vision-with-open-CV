# Experiment 33: Face Detection using OpenCV DNN (SSD ResNet-10 model).
# Number: 33
# Name: Face Detection using OpenCV DNN

import cv2
import numpy as np
import os
import urllib.request

# SSD ResNet-10 face detector (Caffe model) — works reliably with OpenCV 4 & 5
PROTOTXT_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_PATH   = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

def download_model():
    """Download the SSD face detection model files if not already present."""
    for url, path in [(PROTOTXT_URL, PROTOTXT_PATH), (CAFFEMODEL_URL, CAFFEMODEL_PATH)]:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} ...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  Saved to: {path}")
            except Exception as e:
                print(f"  Download failed: {e}")
                return False
    return True

def main():
    image_path  = os.path.join("images", "face_test.jpg")
    output_path = os.path.join("outputs", "Exp33_Face_Detection.png")

    print(f"Reading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}. Run create_test_assets.py first.")
        return

    h, w = img.shape[:2]

    # ── Download model files ──────────────────────────────────────────────────
    if not download_model():
        print("Could not download model files. Aborting.")
        return

    # ── Load DNN model ────────────────────────────────────────────────────────
    print("Loading SSD ResNet-10 face detector...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    # ── Prepare blob and run inference ────────────────────────────────────────
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    # ── Draw bounding boxes for confident detections ──────────────────────────
    result_img  = img.copy()
    face_count  = 0
    CONFIDENCE_THRESHOLD = 0.5      # 50 %

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        face_count += 1
        # Scale box coordinates back to original image size
        box   = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        # Clamp to image bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Face {confidence * 100:.1f}%"
        label_y = max(y1 - 8, 16)
        cv2.putText(result_img, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    print(f"Detected {face_count} face(s) with confidence ≥ {CONFIDENCE_THRESHOLD * 100:.0f}%.")

    cv2.imwrite(output_path, result_img)
    print(f"Result saved to: {output_path}")

    cv2.imshow("Original Image", img)
    cv2.imshow("Face Detection (SSD DNN)", result_img)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
