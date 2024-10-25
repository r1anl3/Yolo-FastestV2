import os
import cv2
import time
import argparse
import csv
import torch
import model.detector
import utils.utils

if __name__ == '__main__':
    # Specify training configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--input_dir', type=str, default='', 
                        help='Directory of input images')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save result images')
    parser.add_argument('--csv_file', type=str, default='results.csv', 
                        help='CSV file to save results')
    
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "Please specify the correct model path"
    assert os.path.exists(opt.input_dir), "Please specify a valid input directory"
    os.makedirs(opt.output_dir, exist_ok=True)  # Create output directory if not exists

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    # Load label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    
    # Prepare CSV file
    with open(opt.csv_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image", "Persons", "Average Confidence"])

        # Process each image in the directory
        for img_file in os.listdir(opt.input_dir):
            img_path = os.path.join(opt.input_dir, img_file)
            ori_img = cv2.imread(img_path)
            if ori_img is None:
                # print(f"Skipping file {img_file} (not an image)")
                continue

            # Data preprocessing
            res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0

            # Model inference
            start = time.perf_counter()
            preds = model(img)
            end = time.perf_counter()
            inference_time = (end - start) * 1000.0
            print(f"{img_file} - Forward time: {inference_time:.2f} ms")

            # Post-process predictions
            output = utils.utils.handel_preds(preds, cfg, device)
            output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

            h, w, _ = ori_img.shape
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]

            person_count = 0
            confidence_scores = []

            # Draw bounding boxes and collect predictions for persons
            for box in output_boxes[0]:
                box = box.tolist()
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]
                
                # Check if the detected object is a person
                if category.lower() == "person":
                    person_count += 1
                    confidence_scores.append(obj_score)
                    
                    x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                    x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(ori_img, f'{obj_score:.2f}', (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
                    cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

            # Calculate average confidence score
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            # Write result to CSV
            csv_writer.writerow([img_file, person_count, f"{avg_confidence:.2f}"])

            # Save result image
            output_img_path = os.path.join(opt.output_dir, f"{os.path.splitext(img_file)[0]}_result.png")
            cv2.imwrite(output_img_path, ori_img)

    print("Processing complete. Results saved in the specified output directory and CSV file.")
