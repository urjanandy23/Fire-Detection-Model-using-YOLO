# ==============================
# Train Fire Detection Model
# ==============================

from ultralytics import YOLO
import sys
import os

# Allow imports from src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *

def train_model():
    print("Loading pretrained model...")
    model = YOLO(PRETRAINED_MODEL)

    print("Starting training...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE
    )

    print("Training completed!")

if __name__ == "__main__":
    train_model()
