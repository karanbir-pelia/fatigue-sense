import cv2
import mediapipe as mp
import logging
import time
import math

class PostureAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize MediaPipe Pose model for upper body posture detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Thresholds
        self.HEAD_TILT_THRESHOLD = 4  # Angle threshold for head tilt
        self.SHOULDER_MISALIGNMENT_THRESHOLD = 3  # Angle threshold for shoulder misalignment
        self.POSTURE_ALERT_THRESHOLD = 1.5  # Time in seconds for posture alert
        self.posture_start_time = None  # Timer for posture alert

        # Store last pose result for other modules to use
        self.last_pose_result = None

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_angle(self, point1, point2):
        """Calculate the angle between two points relative to the horizontal axis."""
        delta_y = point2[1] - point1[1]
        delta_x = point2[0] - point1[0]
        angle = math.atan2(delta_y, delta_x) * 180.0 / math.pi
        return angle

    def get_last_pose_result(self):
        """Return the most recent pose detection result."""
        return self.last_pose_result

    def detect_posture(self, frame):
        """Detect upper body posture using MediaPipe Pose keypoints."""
        person_visible = False

        try:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            alerts = []

            # Process the frame and get pose landmarks
            result_pose = self.pose.process(rgb_frame)
            self.last_pose_result = result_pose  # Store for other modules

            if not result_pose.pose_landmarks:
                return [], person_visible

            person_visible = True
            # Extract key points for the upper body (shoulders, head)
            landmarks = result_pose.pose_landmarks.landmark
            shoulder_left = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            head = landmarks[self.mp_pose.PoseLandmark.NOSE]  # Using the nose as a proxy for the head
            elbow_left = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            elbow_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist_left = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            wrist_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            # Draw the key points on the frame
            cv2.circle(frame, (int(shoulder_left.x * frame.shape[1]), int(shoulder_left.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(shoulder_right.x * frame.shape[1]), int(shoulder_right.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(head.x * frame.shape[1]), int(head.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(elbow_left.x * frame.shape[1]), int(elbow_left.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(elbow_right.x * frame.shape[1]), int(elbow_right.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(wrist_left.x * frame.shape[1]), int(wrist_left.y * frame.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(wrist_right.x * frame.shape[1]), int(wrist_right.y * frame.shape[0])), 5, (0, 255, 0), -1)

            # Connect the landmarks with lines for better visualization
            # Connect the shoulders with a line
            cv2.line(frame,
                    (int(shoulder_left.x * frame.shape[1]), int(shoulder_left.y * frame.shape[0])),
                    (int(shoulder_right.x * frame.shape[1]), int(shoulder_right.y * frame.shape[0])),
                    (0, 255, 0), 2)
            # Shoulder to Elbow
            cv2.line(frame, (int(shoulder_left.x * frame.shape[1]), int(shoulder_left.y * frame.shape[0])),
                     (int(elbow_left.x * frame.shape[1]), int(elbow_left.y * frame.shape[0])), (0, 255, 0), 2)
            cv2.line(frame, (int(shoulder_right.x * frame.shape[1]), int(shoulder_right.y * frame.shape[0])),
                     (int(elbow_right.x * frame.shape[1]), int(elbow_right.y * frame.shape[0])), (0, 255, 0), 2)
            # Elbow to Wrist
            cv2.line(frame, (int(elbow_left.x * frame.shape[1]), int(elbow_left.y * frame.shape[0])),
                     (int(wrist_left.x * frame.shape[1]), int(wrist_left.y * frame.shape[0])), (0, 255, 0), 2)
            cv2.line(frame, (int(elbow_right.x * frame.shape[1]), int(elbow_right.y * frame.shape[0])),
                     (int(wrist_right.x * frame.shape[1]), int(wrist_right.y * frame.shape[0])), (0, 255, 0), 2)

            # Calculate midpoint of shoulders
            shoulder_mid_x = (shoulder_left.x + shoulder_right.x) / 2
            shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
            shoulder_mid = (int(shoulder_mid_x * frame.shape[1]), int(shoulder_mid_y * frame.shape[0]))

            # Draw a line from the midpoint of the shoulders to the nose (head)
            cv2.line(frame,
                    shoulder_mid,
                    (int(head.x * frame.shape[1]), int(head.y * frame.shape[0])),
                    (100, 100, 200), 2)  # Yellow line from midpoint of shoulders to head

            y_offset = 30  # Initial position for drawing text

            # Calculate distance from the nose to the midpoint of the shoulders
            nose_to_midpoint_distance = math.sqrt(
                (head.x * frame.shape[1] - shoulder_mid[0])**2 + (head.y * frame.shape[0] - shoulder_mid[1])**2
            )

            # Calculate distance from the midpoint of the shoulders to either shoulder
            midpoint_to_shoulder_distance = math.sqrt(
                (shoulder_mid[0] - int(shoulder_left.x * frame.shape[1]))**2 + (shoulder_mid[1] - int(shoulder_left.y * frame.shape[0]))**2
            )

            # Calculate angle of shoulders relative to horizontal axis
            shoulder_angle = abs(180 - abs(self.calculate_angle(
                (shoulder_left.x * frame.shape[1], shoulder_left.y * frame.shape[0]),
                (shoulder_right.x * frame.shape[1], shoulder_right.y * frame.shape[0])
            )))

            cv2.putText(frame, f"Shoulder alignment angle: {"%.2f"%shoulder_angle}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30  # Move to next line for the next alert

            # Threshold for misalignment (adjust as needed)
            if abs(shoulder_angle) > self.SHOULDER_MISALIGNMENT_THRESHOLD:
                alerts.append("shoulder misalignment")

            # Calculate the deviation in shoulder alignment
            shoulder_mid = ((shoulder_left.x + shoulder_right.x) / 2, (shoulder_left.y + shoulder_right.y) / 2)
            head_mid = (head.x, head.y)

            head_tilt_angle = abs(90 - abs(self.calculate_angle(shoulder_mid, head_mid)))
            cv2.putText(frame, f"Head tilt angle: {"%.2f"%head_tilt_angle}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if the distance from the nose to midpoint is less than 70% of the distance to the shoulder
            if nose_to_midpoint_distance < 0.7 * midpoint_to_shoulder_distance:
                alerts.append("neck posture")

            # If the angle is above the threshold, alert for head tilt
            if head_tilt_angle > self.HEAD_TILT_THRESHOLD:
                alerts.append("head tilt")

            # Handle posture timer logic
            if alerts:
                if self.posture_start_time is None:
                    self.posture_start_time = time.time()  # Start timer when posture issue is detected
                elif time.time() - self.posture_start_time >= self.POSTURE_ALERT_THRESHOLD:
                    return alerts, person_visible  # Only return alerts if sustained for threshold time
                else:
                    return [], person_visible  # Issues detected but not sustained long enough
            else:
                self.posture_start_time = None  # Reset posture timer if no issue detected
                return [], person_visible

        except Exception as e:
            self.logger.error(f"Error in posture detection: {str(e)}")
            return [], person_visible

    def close(self):
        """Clean up resources."""
        if self.pose:
            self.pose.close()