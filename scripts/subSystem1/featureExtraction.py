import cv2
from feat import Detector
from PIL import Image
import os
import torch

detector = Detector(device='cuda' if torch.cuda.is_available() else 'cpu')

TEMP_IMAGE_PATH = "temp_frame.jpg"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

iteration = -1

while True:

    iteration += 1

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow("Webcam", frame)

    # Only analyze every nth frame (performance tradeoff)
    if iteration % 30 == 0:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # This is a workaround as the detect_image function doesn't accept numpy arrays
        pil_image = Image.fromarray(rgb_frame)
        pil_image.save(TEMP_IMAGE_PATH)

        try:

            features = detector.detect_image(TEMP_IMAGE_PATH)

            # At this point, the classifier function should be called
            # For now:
            if not features.empty:
                print("Action Units:", features.aus.to_dict())
            else:
                print("No face detected.")
        except Exception as e:
            print("Error analyzing frame:", e)

    if cv2.waitKey(1) == ord('q'):
        break

if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)

cap.release()
cv2.destroyAllWindows()
