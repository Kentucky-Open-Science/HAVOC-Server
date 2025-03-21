import cv2
import numpy as np
from yolo_fall_detection import FallDetector


# def main():
#     detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt")
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Webcam could not be opened")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame")
#             break

#         # Process frame using your FallDetector class
#         processed_frame = detector.test_process_frame_pose_fall(frame)

#         cv2.imshow('Fall Detection Webcam Test', processed_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



def main():
    image_path = "static/fallen_man.jfif"
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Resize image to make it larger for readability
    scale_factor = 2.5
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Change white background to dark blue
    white_threshold = 200
    mask = cv2.inRange(img, (white_threshold, white_threshold, white_threshold), (255, 255, 255))
    img[mask == 255] = (100, 0, 50)  # Dark blue color

    detector = FallDetector()
    processed_frame = detector.test_process_frame_pose_fall(img)

    cv2.imshow('Fall Detection Test', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()