import cv2
import logging

class Visualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def draw_face_status(self, frame, fatigue_alert, posture_alerts):
        """Draw face-related status information on the frame."""
        y_offset = 120 + 30 * len(posture_alerts)
        if fatigue_alert:
            cv2.putText(frame, "FATIGUE ALERT!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def draw_posture_status(self, frame, posture_alerts):
        """Draw posture-related status information on the frame."""
        y_offset = 90
        if posture_alerts:
            for alert in posture_alerts:
                cv2.putText(frame, f"{alert.upper()} ALERT!", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        return frame

    def draw_hands_status(self, frame, hands_info, posture_alerts):
        """Draw hands-related status information on the frame."""
        if hands_info and hands_info.get("hands_off_wheel_alert", False):
            # stack immediately below any posture alerts
            y_offset = 90 + 30 * len(posture_alerts)
            cv2.putText(frame,
                        "HANDS OFF WHEEL ALERT!",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def draw_summary(self, frame, fatigue_alert, posture_alerts, person_visible, hands_info):
        """Draw overall status summary on the frame."""
        hands_alert = hands_info.get("hands_off_wheel_alert", False) if hands_info else False
        # If no issues detected and person is visible, show "ALL GOOD"
        if person_visible and not any([fatigue_alert, posture_alerts, hands_alert]):
            cv2.putText(frame, "ALL GOOD!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
