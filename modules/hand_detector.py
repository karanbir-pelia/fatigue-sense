import cv2
import mediapipe as mp
import time
import logging
import math

class HandDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize MediaPipe Hands model for hand landmark detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,  # Detect up to 2 hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Thresholds
        self.HAND_ON_WHEEL_THRESHOLD = 3  # Time in seconds for hands on wheel alert
        self.hands_off_wheel_start_time = None  # Timer for hands off wheel alert

        # Define the hand landmark connections (using MediaPipe's predefined connections)
        self.HAND_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky finger
        ]

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def detect_hands(self, frame, pose_result=None):
        """Detect hands and check if they are on the steering wheel."""
        try:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Define a region for the steering wheel (approximate location on the frame)
            steering_wheel_center = (frame.shape[1] // 2, frame.shape[0] // 2 + 175)  # Center of steering wheel area
            steering_wheel_radius = 175  # Radius of the region considered as the "steering wheel"

            # Draw the steering wheel region
            cv2.circle(frame, steering_wheel_center, steering_wheel_radius, (255, 255, 0), 2)

            # Process the frame and get hand landmarks
            result_hands = self.hands.process(rgb_frame)

            left_hand_on_wheel = False
            right_hand_on_wheel = False

            if result_hands.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result_hands.multi_hand_landmarks):
                    # Iterate over the hand landmarks to draw them
                    for i in range(len(hand_landmarks.landmark)):
                        hand_landmark = hand_landmarks.landmark[i]
                        x = int(hand_landmark.x * frame.shape[1])
                        y = int(hand_landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw the landmark points

                    # Connect the landmarks using predefined connections
                    for connection in self.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = hand_landmarks.landmark[start_idx]
                        end = hand_landmarks.landmark[end_idx]
                        start_x, start_y = int(start.x * frame.shape[1]), int(start.y * frame.shape[0])
                        end_x, end_y = int(end.x * frame.shape[1]), int(end.y * frame.shape[0])
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (150, 150, 150), 2)

                    # Check the distance from the index finger tip (landmark index 8) to the steering wheel center
                    index_finger_tip = hand_landmarks.landmark[8]
                    index_finger_x = int(index_finger_tip.x * frame.shape[1])
                    index_finger_y = int(index_finger_tip.y * frame.shape[0])

                    # Calculate distance from the index finger to the steering wheel center
                    distance_from_wheel = self.calculate_distance(
                        (index_finger_x, index_finger_y),
                        steering_wheel_center
                    )

                    # Check if the hand is within the steering wheel region
                    if distance_from_wheel < steering_wheel_radius:
                        # The hand is on the steering wheel
                        if hand_idx == 0:  # First detected hand
                            left_hand_on_wheel = True
                        else:
                            right_hand_on_wheel = True

            # Check for hands off wheel for more than threshold seconds
            hands_off_wheel_alert = False
            if not left_hand_on_wheel or not right_hand_on_wheel:
                if self.hands_off_wheel_start_time is None:
                    self.hands_off_wheel_start_time = time.time()
                elif time.time() - self.hands_off_wheel_start_time >= self.HAND_ON_WHEEL_THRESHOLD:
                    hands_off_wheel_alert = True
            else:
                self.hands_off_wheel_start_time = None  # Reset timer if hands are on the wheel

            return {
                "left_hand_on_wheel": left_hand_on_wheel,
                "right_hand_on_wheel": right_hand_on_wheel,
                "hands_off_wheel_alert": hands_off_wheel_alert
            }

        except Exception as e:
            self.logger.error(f"Error in hand detection: {str(e)}")
            return {
                "left_hand_on_wheel": False,
                "right_hand_on_wheel": False,
                "hands_off_wheel_alert": False
            }

    def close(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()