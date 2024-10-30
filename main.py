import cv2
import torch
import numpy as np
from torchvision import transforms

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
model = torch.load(model_path)
model.eval()

# Define image pre-processing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust size according to model requirements
    transforms.ToTensor(),
])

# Open CSI camera
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Pre-process the frame for the model
        input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            count = int(output.item())  # Assume output is the human count

        # Display the result
        cv2.putText(frame, f'Human Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('CSI Camera - Human Counting', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
