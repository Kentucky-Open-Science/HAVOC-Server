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
        self.keypoints_labels = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]

    def test_process_frame_box(self, img):
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
    
    def test_process_frame_pose(self, img):
        """
        Processes a single frame and plots pose keypoints detected by YOLO.

        :param img: The input frame (numpy array).
        :return: Frame with keypoints plotted.
        """
        results = self.model(img, stream=True, conf=self.conf_threshold)

        # Iterate over the detection results
        for result in results:
            keypoints = result.keypoints

            if keypoints is not None:
                for person_keypoints in keypoints.xy:
                    for idx, keypoint in enumerate(person_keypoints):
                        x, y = map(int, keypoint[:2])
                        # Plot keypoints on the image
                        cv2.circle(img, (x, y), 5, self.UKBlue, -1)

                        # # Optionally label keypoints
                        # joint_name = [
                        #     "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
                        #     "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                        #     "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
                        #     "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
                        # ][idx]
                        # cv2.putText(img, joint_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.UKBlue, 1)

        return img
    
    def test_process_frame_pose_fall(self, img):
        results = self.model(img, stream=True, conf=self.conf_threshold)

        for r in results:
            for keypoints in r.keypoints.xy:
                points = keypoints.cpu().numpy()
                if len(points) < 17:
                    continue  # Skip if insufficient keypoints

                # Plot keypoints
                for idx, (x, y) in enumerate(points):
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    
                    # cv2.putText(img, self.keypoints_labels[idx], (int(x) + 5, int(y) - 5), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Fall detection logic with error handling
                required_points_indices = [5, 6, 11, 12, 15, 16]
                if len(points) <= max(required_points_indices) or any(points[idx][0] == 0 and points[idx][1] == 0 for idx in required_points_indices):
                    continue  # Skip detection if critical points are missing or out of bounds

                required_points = {
                    "LShoulder": points[5],
                    "RShoulder": points[6],
                    "LHip": points[11],
                    "RHip": points[12],
                    "LAnkle": points[15],
                    "RAnkle": points[16]
                }

                shoulder_avg = (required_points["LShoulder"] + required_points["RShoulder"]) / 2
                hip_avg = (required_points["LHip"] + required_points["RHip"]) / 2

                # Vertical distances (y-axis)
                dist_shoulder_ankle_L = abs(required_points["LAnkle"][1] - required_points["LShoulder"][1])
                dist_shoulder_ankle_R = abs(required_points["RAnkle"][1] - required_points["RShoulder"][1])

                # True (Euclidean) distance between shoulders and hips
                dist_shoulder_hip = math.sqrt((shoulder_avg[0] - hip_avg[0])**2 + (shoulder_avg[1] - hip_avg[1])**2)

                # Visualization lines
                cv2.line(img, tuple(shoulder_avg.astype(int)), tuple(hip_avg.astype(int)), (255, 255, 0), 2)
                cv2.putText(img, f"S-H: {dist_shoulder_hip:.1f}", (int(shoulder_avg[0]), int(shoulder_avg[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


                # #line from ankle to shoulder
                # cv2.line(img, tuple(required_points["LShoulder"].astype(int)), tuple(required_points["LAnkle"].astype(int)), (0, 255, 255), 2)
                # cv2.line(img, tuple(required_points["RShoulder"].astype(int)), tuple(required_points["RAnkle"].astype(int)), (0, 255, 255), 2)

                #line from ankle to vertical distance up calculated above to shoulder
                cv2.line(img, tuple(required_points["LAnkle"].astype(int)), (int(required_points["LAnkle"][0]), int(required_points["LAnkle"][1] - dist_shoulder_ankle_L)), (0, 255, 255), 2)
                cv2.line(img, tuple(required_points["RAnkle"].astype(int)), (int(required_points["RAnkle"][0]), int(required_points["RAnkle"][1] - dist_shoulder_ankle_R)), (0, 255, 255), 2)



                cv2.putText(img, f"LS-LA: {dist_shoulder_ankle_L:.1f}", (int(required_points["LShoulder"][0]), int(required_points["LShoulder"][1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(img, f"RS-RA: {dist_shoulder_ankle_R:.1f}", (int(required_points["RShoulder"][0]), int(required_points["RShoulder"][1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Determine fall
                if (dist_shoulder_ankle_L < dist_shoulder_hip) or (dist_shoulder_ankle_R < dist_shoulder_hip):
                    x1, y1 = int(shoulder_avg[0]), int(shoulder_avg[1])
                    cv2.putText(img, "Fall Detected", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        return img



    def reset(self):
        # Reset any internal state, clear caches, etc.
        pass