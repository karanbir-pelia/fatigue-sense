import pygame
import logging

class AlertSystem:
    def __init__(self, alert_sound_path):
        self.logger = logging.getLogger(__name__)

        # Initialize pygame mixer for sound
        pygame.mixer.init()

        # Load the alert sound
        try:
            self.alert_sound = pygame.mixer.Sound(alert_sound_path)
            self.logger.info("Alert sound loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading alert sound: {str(e)}")
            raise

    def play_alert(self):
        """Play alert sound if it's not already playing."""
        if not pygame.mixer.get_busy():
            self.alert_sound.play()