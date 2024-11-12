import cv2
import torch
import numpy as np
import time
import utils.utils
import model.detector

# Define the CSI camera pipeline
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=15,
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

if __name__ == '__main__':
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/coco2017-0.241078ap-model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Start CSI camera capture
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open CSI camera")
        exit()

    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to model input size
        res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0

        # Model inference
        start = time.perf_counter()
        preds = model(img)
        end = time.perf_counter()
        print(f"Inference time: {(end - start) * 1000:.2f} ms")

        # Process predictions
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

        h, w, _ = frame.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]
        person_count = 0
        # Draw bounding boxes
        for box in output_boxes[0]:
            box = box.tolist()
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            #if category == 'person' and obj_score > 0.8:
            if category == 'person':
                person_count += 1
                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f'{category} {obj_score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Total: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display result
        cv2.imshow("CSI Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
