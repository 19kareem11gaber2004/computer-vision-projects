import cv2
from PIL import Image
from uti import get_limits
import numpy as np


colors_bgr = {
    "Yellow": (0, 255, 255),
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0)
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, bgr in colors_bgr.items():
        lower, upper = get_limits(bgr)
        mask = cv2.inRange(hsv_image, lower, upper)

        pil_mask = Image.fromarray(mask)
        bbox = pil_mask.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, f"{color_name} Object", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    cv2.imshow('Detected Colors', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
