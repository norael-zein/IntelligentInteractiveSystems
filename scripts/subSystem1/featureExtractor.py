import os
import cv2
import time
import torch
import threading
from PIL import Image
from feat import Detector

TEMP_IMAGE_PATH = "temp_frame.jpg"

class FeatureExtractor:

    def __init__(self):
        """
        Initializes the feature extraction subsystem.
        Spawns a thread to continuously capture and process the webcam data with a rate of at most 1 Hz.
        Raises an exception if the webcam can't be opened.
        To avoid memory leaks, make sure to call clean_up() at the program end.
        """

        self.__detector = Detector(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.__cap = cv2.VideoCapture(0)
        self.__lock = threading.Lock()
        self.__latest_aus = None
        self.__face_data = (0.0, 0.0, 0.0, 0.0)
        self.__stop_event = threading.Event()

        if not self.__cap.isOpened():
            raise Exception("Error (FeatureExtraction): Could not open webcam")

        # Thread for capturing and processing frames continuously
        self.__thread = threading.Thread(target=self.__capture_and_extract, daemon=True)
        self.__thread.start()

        # Get width and height of the screen
        self.__screen_bounds = (self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __capture_and_extract(self):
        """
        Continuously captures frames and extracts action units and face data if available.
        Skips frames if the previous one is still being processed or the rate goes below 1 Hz.
        """

        while not self.__stop_event.is_set():

            # Record start time
            start_time = time.time()

            ret, frame = self.__cap.read()

            if not ret:
                print("Error (FeatureExtraction): Failed to capture frame")
                continue

            # Convert frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save frame as a temporary image
            pil_image = Image.fromarray(rgb_frame)
            pil_image.save(TEMP_IMAGE_PATH)

            try:
                # Perform feature detection
                features = self.__detector.detect_image([TEMP_IMAGE_PATH])

                # Set new AUs if available
                aus = features.aus.dropna() if not features.aus.dropna().empty else None

                # Calculate face data if available
                if features.faceboxes.dropna().empty:
                    measurements = self.__face_data
                else:
                    face = features.faceboxes.dropna()
                    measurements = face.iat[0, 0], face.iat[0, 1], face.iat[0, 2], face.iat[0, 3]

            except Exception as e:
                print("Error (FeatureExtraction): {}".format(e))
                aus = None
                measurements = self.__face_data

            # Update the results safely
            with self.__lock:
                self.__latest_aus = aus
                self.__face_data = measurements

            # Ensure at least 1 second has passed before processing the next frame
            elapsed_time = time.time() - start_time
            if elapsed_time < 1.0:
                time.sleep(1.0 - elapsed_time)

    def get_action_units(self):
        """
        Returns the latest analyzed action units (AUs) as a pandas DataFrame.
        Returns None if no results are available.
        """
        with self.__lock:
            return self.__latest_aus

    def get_face_data(self):
        """
        Returns the latest face data as a 6-tuple (x, y, w, h, screen_width, screen_height).
        Returns None if no data is available.
        """
        with self.__lock:
            return self.__face_data + self.__screen_bounds

    def clean_up(self):
        """
        Stops and joins the processing thread.
        Releases the webcam and removes temp files.
        """
        # Signal the thread to stop
        self.__stop_event.set()
        # Wait for the thread to finish
        self.__thread.join()
        # Release webcam
        self.__cap.release()

        # Remove temporary file if necessary
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
