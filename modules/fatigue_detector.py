import cv2
import numpy as np
import time
import logging
from tensorflow.keras.models import load_model

class FatigueDetector:
    def __init__(self, model_path):
        self.logger = logging.getLogger(__name__)

        # Load the fatigue detection model
        try:
            self.fatigue_model = load_model(model_path)
            self.logger.info("Fatigue detection model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading fatigue model: {str(e)}")
            raise

        # Thresholds
        self.EYE_CLOSED_THRESHOLD = 2  # Time in seconds for eye closure
        self.eye_closed_start_time = None  # Timer for closed eyes

    def preprocess_for_model(self, frame):
        """Preprocess the frame for the Keras model."""
        try:
            # Step 1: Resize the frame to the input size expected by the model
            processed_frame = cv2.resize(frame, (224, 224))

            # Step 2: Convert BGR to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Step 3: Normalize pixel values
            processed_frame = processed_frame / 255.0  # Normalize to [0, 1]

            # Step 4: Add batch dimension
            processed_frame = np.expand_dims(processed_frame, axis=0)

            return processed_frame  # Shape: (1, 224, 224, 3)
        except Exception as e:
            self.logger.error(f"Error in preprocessing for model: {str(e)}")
            return None

    def predict_fatigue(self, frame):
        """Use the Keras model to predict fatigue level."""
        preprocessed_frame = self.preprocess_for_model(frame)
        if preprocessed_frame is None:
            return None

        try:
            prediction = self.fatigue_model.predict(preprocessed_frame)
            fatigue_class = np.argmax(prediction, axis=1)[0]  # Get class: 0 (active) or 1 (fatigue)
            confidence_score = prediction[0][fatigue_class]  # Get confidence score
            return confidence_score
        except Exception as e:
            self.logger.error(f"Error during fatigue prediction: {str(e)}")
            return None

    def check_fatigue_alert(self, confidence_score):
        """Determine if a fatigue alert should be triggered based on sustained detection."""
        if confidence_score is None:
            return False

        # Check if the user is showing signs of fatigue
        is_fatigued = confidence_score >= 0.70

        # Implement the eye-closed threshold (only alert after sustained detection)
        if is_fatigued:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = time.time()
            elif time.time() - self.eye_closed_start_time >= self.EYE_CLOSED_THRESHOLD:
                return True
        else:
            self.eye_closed_start_time = None  # Reset timer if eyes are open

        return False