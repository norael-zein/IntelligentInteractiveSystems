import cv2
from feat import Detector
from PIL import Image
import os
import torch

TEMP_IMAGE_PATH = "temp_frame.jpg"

class FeatureExtractor:

    def __init__(self):
        """
        Raises an exception if the webcam can't be opened
        """

        self.detector = Detector(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise Exception("Error (FeatureExtraction): Could not open webcam")

    def extract_action_units(self):
        """
        Returns the AUs found in the image as a pandas data frame
        Returns None if no face is detected or an error occurs
        """

        ret, frame = self.cap.read()

        if not ret:
            print("Error (FeatureExtraction): Failed to capture frame")
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb_frame)
        pil_image.save(TEMP_IMAGE_PATH)

        try:
            features = self.detector.detect_image([TEMP_IMAGE_PATH])
            if not features.aus.empty:
                return features.aus
            else:
                print("Error (FeatureExtraction): No face detected")
                return None
        except Exception as e:
            print("Error (FeatureExtraction): {}".format(e))
            return None

    def clean_up(self):
        """
        Call this function before program termination
        to ensure that the webcam is released and temp files are removed
        """

        self.cap.release()

        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)