import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import argparse
import os

# Argument parser
argparser = argparse.ArgumentParser()
argparser.add_argument("--mode", default="img", help="Mode: 'img' or 'video'")
argparser.add_argument("--filePath", default="face.jpg", help="Path to image or video file")
args = argparser.parse_args()

# Process function
def process(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detection.process(img_rgb)

    if result.detections is not None:
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            h_img, w_img, _ = img.shape
            x = int(bbox.xmin * w_img)
            y = int(bbox.ymin * h_img)
            w = int(bbox.width * w_img)
            h = int(bbox.height * h_img)

            # Draw bounding box
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Blur face area
            img[y:y + h, x:x + w] = cv2.blur(img[y:y + h, x:x + w], (30, 30))
    return img

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection

if args.mode == "img":
    img = cv2.imread(args.filePath)
    if img is None:
        raise ValueError(f"Image not found at path: {args.filePath}")

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        result_img = process(img, face_detection)

    cv2.imwrite('face_blurred.jpg', result_img)
    cv2.imshow("Blurred Face", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    raise NotImplementedError("Only image mode is implemented for now.")
