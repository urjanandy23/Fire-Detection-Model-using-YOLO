# 🔥 Fire Detection - Production Ready Script
# Supports training and inference using YOLOv8

import os
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "datasets/fire_dataset/data.yaml"
MODEL_SAVE_DIR = "model"
RESULTS_DIR = "results"
VIDEO_INPUT = "input.mp4"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# TRAINING FUNCTION
# ==============================

def train_model():
    print("[INFO] Starting training...")
    
    model = YOLO("yolov8n.pt")  # pretrained model

    results = model.train(
        data=DATASET_PATH,
        epochs=50,
        imgsz=640,
        project=RESULTS_DIR,
        name="train"
    )

    # Save best model
    best_model_path = os.path.join(RESULTS_DIR, "train", "weights", "best.pt")
    if os.path.exists(best_model_path):
        os.system(f"cp {best_model_path} {MODEL_SAVE_DIR}/best.pt")
        print(f"[INFO] Model saved to {MODEL_SAVE_DIR}/best.pt")

    return results

# ==============================
# IMAGE INFERENCE
# ==============================

def predict_image(image_path):
    print(f"[INFO] Running inference on image: {image_path}")
    
    model = YOLO(os.path.join(MODEL_SAVE_DIR, "best.pt"))

    results = model.predict(
        source=image_path,
        save=True,
        project=RESULTS_DIR,
        name="predict"
    )

    return results

# ==============================
# VIDEO INFERENCE
# ==============================

def predict_video(video_path=VIDEO_INPUT):
    print(f"[INFO] Running inference on video: {video_path}")
    
    model = YOLO(os.path.join(MODEL_SAVE_DIR, "best.pt"))

    results = model.predict(
        source=video_path,
        save=True,
        project=RESULTS_DIR,
        name="video"
    )

    return results

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="train / image / video")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to image or video")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()

    elif args.mode == "image":
        if not args.input:
            raise ValueError("Provide image path using --input")
        predict_image(args.input)

    elif args.mode == "video":
        predict_video(args.input if args.input else VIDEO_INPUT)

    else:
        print("Invalid mode. Use: train / image / video")

