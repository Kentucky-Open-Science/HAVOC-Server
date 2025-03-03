import cv2
import math
from ultralytics import YOLO


class FallDetector:
    """A class for detecting people and identifying potential falls using YOLO."""

    def __init__(self, model_path="yolo_weights/yolo11n-pose.pt", conf_threshold=0.3):
        """
        Initializes the FallDetector class.

        :param model_path: Path to the YOLO pose detection model.
        :param conf_threshold: Confidence threshold for detecting people.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.UKBlue = (0, 51, 160)  # Bounding box color
        self.classNames = ["person"]  # Only tracking people

    def process_frame(self, img):
        """Processes a single frame, detects people, and labels falls."""
        results = self.model(img, stream=True, conf=self.conf_threshold)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                if int(box.cls[0]) == 0:  # Only detect "person"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), self.UKBlue, 3)

                    # Display label and confidence
                    label = f"{self.classNames[0]} {confidence:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.UKBlue, 2)

                    height = y2 - y1
                    width = x2 - x1

                    # Check if fall is detected
                    if height - width < 0:
                        cv2.putText(img, "Fall Detected", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return img  # Return the processed frame
