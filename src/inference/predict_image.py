# ==============================
# Fire Detection on Image
# ==============================

from ultralytics import YOLO
import sys
import os

# Allow imports from src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *

def predict_image(image_path):
    print("Loading trained model...")
    model = YOLO(TRAINED_MODEL)

    print(f"Running detection on {image_path} ...")
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE,
        iou=IOU,
        save=SAVE_RESULTS
    )

    print("Prediction completed!")

if __name__ == "__main__":
    # Change this to your test image path
    predict_image("test.jpg")
