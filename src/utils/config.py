# ==============================
# Configuration File
# ==============================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_YAML = os.path.join(BASE_DIR, "data", "processed", "data.yaml")
VIDEO_DIR = os.path.join(BASE_DIR, "data", "videos")

# Model paths
PRETRAINED_MODEL = os.path.join(BASE_DIR, "models", "pretrained", "yolov8s.pt")
TRAINED_MODEL = os.path.join(BASE_DIR, "models", "trained", "best.pt")

# Training settings
IMAGE_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16

# Inference settings
CONFIDENCE = 0.5
IOU = 0.45
SAVE = True
