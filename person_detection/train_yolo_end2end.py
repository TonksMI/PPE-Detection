"""
End-to-End YOLOv8 Training for PPE Detection
"""
import os
import time
from ultralytics import YOLO

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists("D:/datasets/yolo_ppe_end2end"):
    YOLO_DATA = "D:/datasets/yolo_ppe_end2end/ppe_end2end.yaml"
else:
    YOLO_DATA = os.path.join(BASE, "datasets", "yolo_ppe_end2end", "ppe_end2end.yaml")

def main():
    print("="*65)
    print("STARTING End-to-End YOLO PPE TRAINING")
    print("="*65)
    
    if not os.path.exists(YOLO_DATA):
        print(f"Error: {YOLO_DATA} does not exist.")
        return

    # Initialize model
    model = YOLO('yolov8n.pt') 

    # Train model
    # Using 15 epochs, imgsz 640. YOLO uses single pass.
    t0 = time.time()
    results = model.train(
        data=YOLO_DATA,
        epochs=200,
        imgsz=640,
        batch=32,
        name='yolov8_ppe_e2e_prod',
        patience=5
    )
    
    print(f"Training completed in {(time.time()-t0)/60:.1f} minutes")
    print(f"Metrics: {results}")

if __name__ == "__main__":
    main()
