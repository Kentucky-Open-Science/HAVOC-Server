import cv2
from yolo_fall_detection import FallDetector

def main():
    detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Process frame using your FallDetector class
        processed_frame = detector.test_process_frame_pose(frame)

        cv2.imshow('Fall Detection Webcam Test', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
