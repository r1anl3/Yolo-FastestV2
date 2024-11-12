import cv2
import torch
import numpy as np
import time
import utils.utils

# Define the CSI camera pipeline
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# Load PyTorch model
model_path = 'path_to_your_model.pth'  # Replace with your model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

cfg = { "width": 352, "height": 352, "names": '/data/coco.names' }
LABEL_NAMES = []
with open(cfg["names"], 'r') as f:
    for line in f.readlines():
        LABEL_NAMES.append(line.strip())

# Open CSI camera
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

# Initialize the timer
last_detection_time = time.time()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Only run detection every 2 seconds
        current_time = time.time()
        if current_time - last_detection_time >= 2:
            last_detection_time = current_time  # Update the last detection time

            # Resize and preprocess frame
            res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR) 
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(device).float() / 255.0

            # Perform inference
            with torch.no_grad():
                start = time.perf_counter()
                preds = model(img)
                end = time.perf_counter()
                print(f"Forward time: {(end - start) * 1000.0:.2f} ms")

                output = utils.utils.handel_preds(preds, cfg, device)
                output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

            # Scaling factors
            h, w, _ = frame.shape
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]

            # Initialize counter
            person_count = 0

            for box in output_boxes[0]:
                box = box.tolist()
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                # If category is 'person', count it
                if category == "person":
                    person_count += 1

                # Scale bounding box
                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                # Draw bounding box and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f'{obj_score:.2f}', (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)    
                cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

            # Display person count on frame
            cv2.putText(frame, f'Persons: {person_count}', (10, 30), 0, 1, (0, 0, 255), 2)

        # Display result frame
        cv2.imshow("Person Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
