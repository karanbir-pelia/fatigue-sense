import cv2
import logging
import pygame
from modules.fatigue_detector import FatigueDetector
from modules.posture_analyzer import PostureAnalyzer
from modules.hand_detector import HandDetector
from modules.alert_system import AlertSystem
from modules.visualizer import Visualizer

class DriverMonitoringSystem:
    def __init__(self, fatigue_model_path, alert_sound_path='alert_sound.wav'):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize pygame mixer for sound
        pygame.mixer.init()

        # Initialize component systems
        try:
            self.fatigue_detector = FatigueDetector(fatigue_model_path)
            self.posture_analyzer = PostureAnalyzer()
            self.hand_detector = HandDetector()
            self.alert_system = AlertSystem(alert_sound_path)
            self.visualizer = Visualizer()
            self.logger.info("All modules initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing modules: {str(e)}")
            raise

    def process_frame(self, frame):
        """Process a single frame for all monitoring systems"""
        if frame is None:
            self.logger.warning("Empty frame received")
            return frame

        try:
            frame = cv2.flip(frame, 1)  # Flip frame horizontally for natural view

            # Run fatigue detection
            fatigue_score = self.fatigue_detector.predict_fatigue(frame)
            fatigue_alert = self.fatigue_detector.check_fatigue_alert(fatigue_score)

            # Run posture detection with proper error handling
            result = self.posture_analyzer.detect_posture(frame)
            if result is None:
                posture_issues, person_visible = [], False
            else:
                posture_issues, person_visible = result

            # Run hand position detection
            hands_info = self.hand_detector.detect_hands(frame, self.posture_analyzer.get_last_pose_result())

            # Update the visualizer with current results
            frame = self.visualizer.draw_face_status(frame, fatigue_alert, posture_issues)
            frame = self.visualizer.draw_posture_status(frame, posture_issues)
            frame = self.visualizer.draw_hands_status(frame, hands_info, posture_issues)
            frame = self.visualizer.draw_summary(frame, fatigue_alert, posture_issues, person_visible, hands_info)

            # Play alert sound if needed
            if fatigue_alert or (posture_issues and len(posture_issues) > 0) or (hands_info and hands_info.get("hands_off_wheel_alert", False)):
                self.alert_system.play_alert()

            return frame
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            # Return original frame if there's an error in processing
            return frame

    def run(self, video_source=0):
        """Run the monitoring system on video input"""
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                self.logger.error(f"Cannot open video source {video_source}")
                return

            self.logger.info(f"Starting video capture from source {video_source}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video stream")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display frame
                cv2.imshow('Driver Monitoring System', processed_frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User requested exit (q key)")
                    break

        except Exception as e:
            self.logger.error(f"Error in run loop: {str(e)}")
        finally:
            self.logger.info("Cleaning up resources")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            self.cleanup()

    def cleanup(self):
        """Clean up all resources."""
        try:
            # Close all MediaPipe resources
            self.posture_analyzer.close()
            self.hand_detector.close()
            self.logger.info("All resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Initialize the monitoring system
    try:
        monitoring_system = DriverMonitoringSystem(
            fatigue_model_path='models/best_fatigue_model.keras',
            alert_sound_path='sounds/alert_sound.wav'
        )

        # Run the system
        monitoring_system.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")