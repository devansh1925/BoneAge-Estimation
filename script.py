import os
import shutil
from ultralytics import YOLO
import cv2
from torchvision.ops import nms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import torch
import numpy as np
import segmentation_models_pytorch as smp
import math
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def delete_old_files_and_folders():
   
    ################################################ deleting old generated files. ################################################
    base_path = "workspace"
    delete_folders = {"AngleLines", "BrightJoints","Joints","RotatedJoints", "RotatedJointsEdges", "RotatedJointsMasks", "RotatedJointsLines"}
    delete_files = {"Boxes.png", "distance_ratios.csv", "InitialBoxCoords.txt", "Label.txt", "RotatedBoxCoords.txt", "RotationAngles.csv"}

    # Get all items in current directory
    for item in os.listdir(base_path):
        # Delete specified files
        item_path = os.path.join(base_path, item)
        
        if os.path.isfile(item_path) and item in delete_files:
            os.remove(item_path)

        elif os.path.isdir(item_path) and item in delete_folders:
            shutil.rmtree(item_path)

    print("🗑️ Deleted specified files and folders from workspace/")

def main_function(input_image_path, is_female):

    ################################################ Yolo for bound boxes Labels. ################################################

    # Load YOLOv8 model
    model = YOLO("Resources/best.pt")
    # Parameters
    conf_threshold = 0.25
    iou_threshold = 0.4

    # Load image
    image = cv2.imread(input_image_path)
    height, width = image.shape[:2]

    # Run inference
    results = model(input_image_path, conf=conf_threshold, save=False)
    boxes = results[0].boxes

    if boxes is not None and boxes.shape[0] > 0:
        bboxes = boxes.xyxy  # (x1, y1, x2, y2)
        scores = boxes.conf
        class_ids = boxes.cls

        # Apply NMS
        keep_indices = nms(bboxes, scores, iou_threshold)

        filtered_boxes = bboxes[keep_indices]
        filtered_scores = scores[keep_indices]
        filtered_class_ids = class_ids[keep_indices]

        # Get base name of input image (without extension)
        image_name = os.path.splitext(os.path.basename(input_image_path))[0]

        # Define output file names
        output_image_path = f"workspace/Boxes.png"
        label_file_path = f"workspace/Label.txt"

        with open(label_file_path, "w") as f:
            for box, score, cls_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
                x1, y1, x2, y2 = box.tolist()
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                f.write(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                # Draw on image
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{model.names[int(cls_id)]} {score:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save boxed image
        cv2.imwrite(output_image_path, image)
        print(f"Saved boxed image: {output_image_path} and labels: {label_file_path}")
    else:
        print(f"No boxes detected for {input_image_path}")

    ################################################ Extracting Joints from Label. ################################################

    label_path = 'workspace/Label.txt'
    output_folder = 'workspace/Joints'

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    image = cv2.imread(input_image_path)
    height, width = image.shape[:2]

    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"Label file not found for {input_image_path}, exiting.")
        return -1,-1
    else:
        # Read label file and extract bounding boxes with class 0, 1, or 2
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"Label file is empty for {input_image_path}, exiting.")
            return -1,-1

        # Collect bbox info in a list: (cls, x1, y1, x2, y2)
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines

            cls, x_center, y_center, w, h = map(float, parts)
            cls_int = int(cls)
            if cls_int not in [0, 1, 2]:
                continue

            # Convert normalized coordinates to pixel values
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            # Calculate top-left and bottom-right coordinates
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            bboxes.append((cls_int, x1, y1, x2, y2))

        # Sort bounding boxes by class label (0, 1, 2)
        bboxes.sort(key=lambda x: x[0])

        # Save crops in order 0, then 1, then 2
        image_name = os.path.basename(input_image_path)
        for cls_int, x1, y1, x2, y2 in bboxes:
            crop = image[y1:y2, x1:x2]
            output_name = f"joint{cls_int}.jpg"
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, crop)

    ################################################ Bright joints. ################################################

    # ---------- Load model from checkpoint ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate model architecture (same as training)
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # No ImageNet weights
        in_channels=1,
        classes=1
    )

    # Load trained weights
    checkpoint_path = "Resources/unetpp_bone_segmentation_full.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ---------- Folders ----------
    input_folder = "workspace/Joints"
    output_folder = "workspace/BrightJoints"
    os.makedirs(output_folder, exist_ok=True)

    # ---------- Preprocessing ----------
    def preprocess_image(image_path, image_size=256):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.resize(img, (image_size, image_size))
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor

    # ---------- Inference ----------
    def predict_mask(model, image_tensor, device):
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output)
            mask_pred = (probs > 0.5).float()
        return mask_pred.squeeze().cpu().numpy()

    # ---------- Save Binary Mask ----------
    def save_mask_as_binary_image(mask_pred, save_path):
        binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255  # 0 or 255
        cv2.imwrite(save_path, binary_mask)

    # ---------- Process All Images ----------
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(input_folder, filename)
            image_tensor = preprocess_image(image_path)
            predicted_mask = predict_mask(model, image_tensor, device)

            output_filename = filename.rsplit(".", 1)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            save_mask_as_binary_image(predicted_mask, output_path)

    print("✅ All binary masks (Bright Joints) saved in:", output_folder)

    ################################################ Angle Lines and RotationAngles.csv. ################################################

    # Define rotation function
    def rotate_image(image, angle_deg):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotated

    # Create output folders if they don't exist
    os.makedirs("workspace/AngleLines", exist_ok=True)

    rotation_data = [("ImageName", "RotationAngle")]

    input_folder = "workspace/BrightJoints"
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            imgPath = os.path.join(input_folder, filename) 
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipping unreadable image: {filename}")
                continue

            # --- Edge Detection and Line Fitting ---
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 90)

            height, width = edges.shape
            top_15_percent = int(height * 0.15)
            start_percent = int(height * 0.03)
            difference_between_pixel_number = 10

            # Check if the image is joint2, then skip left 40%
            # if "joint2" in filename:
            #     skip_left_columns = int(width * 0.4)

            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            left_points = []
            right_points = []

            for y in range(start_percent, top_15_percent):
                row = edges[y]
                white_pixels = np.where(row > 0)[0]

                # if "joint2" in filename:
                #     white_pixels = white_pixels[white_pixels >= skip_left_columns]

                if len(white_pixels) > 0:
                    left_x = white_pixels[0]
                    right_x = white_pixels[-1]

                    left_points.append((left_x, y))
                    if abs(right_x - left_x) >= difference_between_pixel_number:
                        right_points.append((right_x, y))

            if len(left_points) == 0 and "joint2" in filename:
                top_30_percent = int(height * 0.3)

                for y in range(top_30_percent):
                    row = edges[y]
                    white_pixels = np.where(row > 0)[0]


                    # if "joint2" in filename:
                    #     white_pixels = white_pixels[white_pixels >= skip_left_columns]

                    if len(white_pixels) > 0:
                        left_x = white_pixels[0]
                        left_points.append((left_x, y))

            # Curved output
            curved_output = output.copy()
            for i in range(1, len(left_points)):
                cv2.line(curved_output, left_points[i - 1], left_points[i], (0, 255, 0), 1)
            for i in range(1, len(right_points)):
                cv2.line(curved_output, right_points[i - 1], right_points[i], (0, 0, 255), 1)

            # Fitted line
            fitted_output = output.copy()
            line_angle = None
            if len(left_points) > 1:
                left_points_np = np.array(left_points).reshape(-1, 1, 2)
                [vx, vy, x0, y0] = cv2.fitLine(left_points_np, cv2.DIST_L2, 0, 0.01, 0.01)

                y1 = 0
                x1 = int(x0 + (y1 - y0) * vx / vy)
                y2 = height - 1
                x2 = int(x0 + (y2 - y0) * vx / vy)

                cv2.line(fitted_output, (x1, y1), (x2, y2), (255, 0, 0), 1)

                def calculate_line_angle(x1, y1, x2, y2):
                    delta_y = y2 - y1
                    delta_x = x2 - x1
                    angle_radians = math.atan2(delta_y, delta_x)
                    angle_degrees = math.degrees(angle_radians)
                    angle_degrees = -angle_degrees
                    if angle_degrees < 0:
                        angle_degrees += 180
                    elif angle_degrees > 180:
                        angle_degrees -= 180
                    return angle_degrees, angle_radians

                line_angle, _ = calculate_line_angle(x1, y1, x2, y2)

            # Plot 1: Edge lines
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(curved_output)
            plt.title("Curved Green and Red Lines (Top 15%)")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(fitted_output)
            if line_angle is not None:
                plt.title(f"Straight Line Fitted (Angle: {line_angle:.1f}°)")
            else:
                line_angle = 80
                plt.title("Straight Line Fitted")
            plt.axis('off')

            angle_plot_path = os.path.join("workspace/AngleLines", filename)
            plt.savefig(angle_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 2: Rotated image
            rotation_angle = 80 - line_angle if line_angle is not None else 0
            rotated_img = rotate_image(img, rotation_angle)
            rotation_data.append((filename, rotation_angle))

            # print(f"Processed: {filename} | Angle: {line_angle:.2f}° | Rotated by: {rotation_angle:.2f}°")
        
    with open("workspace/RotationAngles.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rotation_data)

    print("✅ All lines images saved to AngleLines.")
    print("📄 Rotation angles saved to RotationAngles.csv.")

    ################################################ 4 coordinates of the joints InitialBoxCoords.txt. ################################################

    # === Input paths ===

    label_path = "workspace/Label.txt"
    output_txt_path = "workspace/InitialBoxCoords.txt"

    # === Load image ===
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Could not read image {input_image_path}.")
        exit()

    img_h, img_w = img.shape[:2]
    image_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # === Read label file ===
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        exit()

    with open(label_path, "r") as f:
        lines = f.readlines()

    # === Process and save corner coordinates ===
    with open(output_txt_path, "w") as out:
        for line in lines:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert normalized values to absolute pixel coordinates
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            corners = [
                (int(x1), int(y1)),  # top-left
                (int(x2), int(y1)),  # top-right
                (int(x2), int(y2)),  # bottom-right
                (int(x1), int(y2))   # bottom-left
            ]

            # Write to file
            corner_str = " ".join(f"{x},{y}" for x, y in corners)
            out.write(f"{int(class_id)} {corner_str}\n")

    print(f"✅ Done! Coordinates saved in '{output_txt_path}'.")

    ################################################ RotatedBoxCoords.txt. ################################################

    # === Utility to rotate a point around a center ===
    def rotate_point(px, py, cx, cy, angle_degrees):
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        tx, ty = px - cx, py - cy
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a

        return int(cx + rx), int(cy + ry)

    # === Load rotation angles CSV ===
    df_angles = pd.read_csv("workspace/RotationAngles.csv")

    # === Image & file names ===
    corner_file = "workspace/InitialBoxCoords.txt"
    output_file = "workspace/RotatedBoxCoords.txt"

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Could not read image {input_image_path}.")
        exit()

    h, w = img.shape[:2]

    # === Extract angles for each joint ===
    rotation_rows = df_angles
    rotation_angles = {}
    for _, row in rotation_rows.iterrows():
        joint_id_str = ''.join(filter(str.isdigit, row["ImageName"]))

        if joint_id_str.isdigit():
            class_id = int(joint_id_str)
            rotation_angles[class_id] = float(row["RotationAngle"])

    # === Process the corner coordinates ===
    if not os.path.exists(corner_file):
        print(f"Corner coordinate file not found: {corner_file}")
        exit()

    rotated_lines = []
    with open(corner_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        corners = [tuple(map(int, pt.split(','))) for pt in parts[1:]]

        cx = sum(x for x, y in corners) / 4
        cy = sum(y for x, y in corners) / 4

        if class_id in rotation_angles:
            angle = rotation_angles[class_id]
            rotated_corners = [rotate_point(x, y, cx, cy, angle) for (x, y) in corners]
        else:
            rotated_corners = corners

        # Save rotated coordinates
        corner_str = " ".join(f"{x},{y}" for x, y in rotated_corners)
        rotated_lines.append(f"{class_id} {corner_str}")

        # Draw box and class label on image
        for i in range(4):
            pt1 = rotated_corners[i]
            pt2 = rotated_corners[(i + 1) % 4]
            color = (255, 0, 0) if class_id in rotation_angles else (0, 255, 0)
            cv2.line(img, pt1, pt2, color, 2)
        cv2.putText(img, str(class_id), rotated_corners[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # === Save updated corner file ===
    with open(output_file, "w") as f:
        for line in rotated_lines:
            f.write(line + "\n")

    print(f"✅ Rotated box coordinates saved to: {output_file}")

    ################################################ Rotated Joints ################################################

    # --- Input/output settings for a single image ---
    rotated_corner_file = "workspace/RotatedBoxCoords.txt"
    output_folder = "workspace/RotatedJoints"
    os.makedirs(output_folder, exist_ok=True)

    # --- Load image ---
    original_img = cv2.imread(input_image_path)
    if original_img is None:
        print(f"Unable to read image: {input_image_path}")
        exit()

    # --- Load rotated corner coordinates ---
    if not os.path.exists(rotated_corner_file):
        print(f"Rotated corner file not found: {rotated_corner_file}")
        exit()

    with open(rotated_corner_file, "r") as f:
        rotated_lines = f.readlines()

    # --- Process each rotated box ---
    for line in rotated_lines:
        parts = line.strip().split()
        class_id = int(parts[0])

        # Only process boxes for class IDs 0, 1, 2
        if class_id not in [0, 1, 2]:
            continue

        corners = [tuple(map(float, pt.split(','))) for pt in parts[1:]]
        src_pts = np.array(corners, dtype=np.float32)

        # Estimate box dimensions
        width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
        height = int(np.linalg.norm(src_pts[0] - src_pts[3]))

        if width == 0 or height == 0:
            print(f"Zero dimension box found for class {class_id}, skipping.")
            continue

        # Destination points (aligned rectangle)
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Apply perspective transform to extract upright box
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(original_img, M, (width, height))

        # Save cropped result
        output_path = os.path.join(output_folder, f"joint{class_id}.jpg")
        cv2.imwrite(output_path, warped)
        print(f"✅ Saved: {output_path}")

    ################################################ Rotated Joints Masks. ################################################

    # ---------- Load model from checkpoint ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate model architecture (same as training)
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # No ImageNet weights
        in_channels=1,
        classes=1
    )

    # Load trained weights
    checkpoint_path = "Resources/unetpp_bone_segmentation_full.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ---------- Folders ----------
    input_folder = "workspace/RotatedJoints"
    output_folder = "workspace/RotatedJointsMasks"
    os.makedirs(output_folder, exist_ok=True)

    # ---------- Preprocessing ----------
    def preprocess_image(image_path, image_size=256):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.resize(img, (image_size, image_size))
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor

    # ---------- Inference ----------
    def predict_mask(model, image_tensor, device):
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output)
            mask_pred = (probs > 0.5).float()
        return mask_pred.squeeze().cpu().numpy()

    # ---------- Save Binary Mask ----------
    def save_mask_as_binary_image(mask_pred, save_path):
        binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255  # 0 or 255
        cv2.imwrite(save_path, binary_mask)

    # ---------- Process All Images ----------
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(input_folder, filename)
            image_tensor = preprocess_image(image_path)
            predicted_mask = predict_mask(model, image_tensor, device)

            output_filename = filename.rsplit(".", 1)[0] + "_mask.png"
            output_path = os.path.join(output_folder, output_filename)
            save_mask_as_binary_image(predicted_mask, output_path)

    print("✅ All binary masks saved in:", output_folder)

    ################################################ Rotated Joints Edges. ################################################

    # --- CONFIGURATION ---
    input_folder = 'workspace/RotatedJointsMasks'      # Replace with your folder of original images
    output_folder = 'workspace/RotatedJointsEdges'  # Folder where edge images will be saved

    # --- Ensure output folder exists ---
    os.makedirs(output_folder, exist_ok=True)

    # --- Loop through all images in the input folder ---
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Canny needs grayscale

            if image is None:
                print(f"Failed to load {img_path}, skipping.")
                continue

            # --- Apply Canny Edge Detection ---
            edges = cv2.Canny(image, threshold1=100, threshold2=200)

            # --- Save result ---
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, edges)

            print(f"Saved edge-detected image: {out_path}")

    ################################################ Green Lines and Distance_Ratios.csv. ################################################

    # Config
    input_folder = "workspace/RotatedJointsEdges"
    output_folder = "workspace/RotatedJointsLines"
    TopIgnorePercent = 0.12
    BottomIgnorePercent = 0.34
    smoothing_sigma = 0.4

    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)

    # Prepare list to store ratios for CSV
    all_ratios = []

    # Process each image in the folder
    for image_name in os.listdir(input_folder):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(input_folder, image_name)

        # Load grayscale image
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)

        height, width = binary.shape
        left_limit = int(0.4 * width)
        right_limit = int(0.6 * width)

        left_x, left_y, right_x, right_y = [], [], [], []

        for y in range(height):
            row = binary[y]
            left_indices = np.where(row == 255)[0]
            if left_indices.size > 0:
                x_left = min(left_indices[0], left_limit)
                left_x.append(x_left)
                left_y.append(y)

            right_indices = np.where(row[::-1] == 255)[0]
            if right_indices.size > 0:
                x_right = max(width - right_indices[0] - 1, right_limit)
                right_x.append(x_right)
                right_y.append(y)

        left_x_smooth = gaussian_filter1d(left_x, sigma=smoothing_sigma)
        right_x_smooth = gaussian_filter1d(right_x, sigma=smoothing_sigma)

        inv_left = -np.array(left_x_smooth)
        minima_indices, _ = find_peaks(inv_left, distance=10, prominence=2)
        maxima_indices, _ = find_peaks(right_x_smooth, distance=10, prominence=2)

        red_points = list(zip(np.array(left_y)[minima_indices], left_x_smooth[minima_indices]))
        blue_points = list(zip(np.array(right_y)[maxima_indices], right_x_smooth[maxima_indices]))

        red_points_sorted = sorted(red_points, key=lambda x: x[0])
        blue_points_sorted = sorted(blue_points, key=lambda x: x[0])

        top_ignore = int(TopIgnorePercent * height)
        bottom_ignore = int((1 - BottomIgnorePercent) * height)

        red_filtered = [(y, x) for (y, x) in red_points_sorted if top_ignore <= y <= bottom_ignore]
        blue_filtered = [(y, x) for (y, x) in blue_points_sorted if top_ignore <= y <= bottom_ignore]

        num_lines = min(len(red_filtered), len(blue_filtered))
        # print(f"\n🔢 => Points on Red: {len(red_filtered)}, Blue: {len(blue_filtered)}")
        # print(f"🔗 Drawing {num_lines} non-intersecting lines")

        red_combos = list(combinations(red_filtered, num_lines))
        blue_combos = list(combinations(blue_filtered, num_lines))

        valid_pairs = []

        for reds in red_combos:
            for blues in blue_combos:
                reds_sorted = sorted(reds, key=lambda pt: pt[0])
                blues_sorted = sorted(blues, key=lambda pt: pt[0])
                intersect = any(r[0] > r2[0] and b[0] < b2[0] or r[0] < r2[0] and b[0] > b2[0]
                                for (r, b), (r2, b2) in zip(zip(reds_sorted, blues_sorted), zip(reds_sorted[1:], blues_sorted[1:])))
                if not intersect:
                    valid_pairs.append(list(zip(reds_sorted, blues_sorted)))

        # print(f"✅ Valid Non-Intersecting Combinations: {len(valid_pairs)}")

        def angle(p1, p2):
            dx = p2[1] - p1[1]
            dy = p2[0] - p1[0]
            return abs(np.degrees(np.arctan2(dy, dx)))

        best_theta = float('inf')
        best_combo = None

        for combo in valid_pairs:
            total_theta = sum(angle(r, b) for r, b in combo)
            if total_theta < best_theta:
                best_theta = total_theta
                best_combo = combo

        image_with_lines = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for y, x in red_filtered:
            cv2.circle(image_with_lines, (int(x), int(y)), 4, (0, 0, 255), -1)
        for y, x in blue_filtered:
            cv2.circle(image_with_lines, (int(x), int(y)), 4, (255, 0, 0), -1)

        all_drawn_lines = []

        if best_combo:
            for (r, b) in best_combo:
                pt1 = (int(r[1]), int(r[0]))
                pt2 = (int(b[1]), int(b[0]))
                cv2.line(image_with_lines, pt1, pt2, (0, 255, 0), 2)
                all_drawn_lines.append((r, b))

        combo_red_ys = {int(r[0]) for r, _ in best_combo} if best_combo else set()
        combo_blue_ys = {int(b[0]) for _, b in best_combo} if best_combo else set()
        top_line_y = min(min(combo_red_ys, default=height), min(combo_blue_ys, default=height))

        unmatched_red_above = [(y, x) for (y, x) in red_filtered if int(y) not in combo_red_ys and y < top_line_y]
        unmatched_blue_above = [(y, x) for (y, x) in blue_filtered if int(y) not in combo_blue_ys and y < top_line_y]

        if len(best_combo) == 1:
            base_r, base_b = best_combo[0]
            dy = base_b[0] - base_r[0]
            dx = base_b[1] - base_r[1]
            line_len = np.hypot(dx, dy)

            if len(unmatched_red_above) == 1:
                y_start, x_start = unmatched_red_above[0]
                slope = dy / dx if dx != 0 else 0
                for x_end in range(int(x_start) + 1, width):
                    y_end = int(y_start + slope * (x_end - x_start))
                    if (0 <= y_end < height and binary[y_end, x_end] == 255) or \
                    (0 <= y_end - 1 < height and binary[y_end - 1, x_end] == 255) or \
                    (0 <= y_end + 1 < height and binary[y_end + 1, x_end] == 255):
                        if np.linalg.norm([x_end - x_start, y_end - y_start]) >= line_len / 2:
                            cv2.line(image_with_lines, (int(x_start), int(y_start)), (x_end, y_end), (0, 255, 0), 2)
                            cv2.circle(image_with_lines, (int(x_end), int(y_end)), 4, (255, 0, 0), -1)  # 🔵 Draw blue point

                            # Save new blue point
                            blue_filtered.append((y_end, x_end))  # Add new point to blue_filtered
                            all_drawn_lines.append(((y_start, x_start), (y_end, x_end)))  # Save new line pair
                            break


            elif len(unmatched_blue_above) == 1:
                y_start, x_start = unmatched_blue_above[0]
                slope = dy / dx if dx != 0 else 0
                for x_end in range(int(x_start) - 1, -1, -1):
                    y_end = int(y_start + slope * (x_end - x_start))
                    if (0 <= y_end < height and binary[y_end, x_end] == 255) or \
                    (0 <= y_end - 1 < height and binary[y_end - 1, x_end] == 255) or \
                    (0 <= y_end + 1 < height and binary[y_end + 1, x_end] == 255):
                        if np.linalg.norm([x_end - x_start, y_end - y_start]) >= line_len / 2:
                            cv2.line(image_with_lines, (int(x_start), int(y_start)), (x_end, y_end), (0, 255, 0), 2)
                            cv2.circle(image_with_lines, (int(x_end), int(y_end)), 4, (0, 0, 255), -1)  # 🔴 Draw red point

                            # Save new red point
                            red_filtered.append((y_end, x_end))  # Add new point to red_filtered
                            all_drawn_lines.append(((y_start, x_start), (y_end, x_end)))  # Save new line pair
                            break

        all_drawn_lines_sorted = sorted(all_drawn_lines, key=lambda pair: pair[0][0])
        line_lengths = [np.linalg.norm([p2[1] - p1[1], p2[0] - p1[0]]) for p1, p2 in all_drawn_lines_sorted]

        joint_number = ''.join(filter(str.isdigit, image_name))  # Extract 0, 1, 2
        box_number = f"Box{joint_number}"  # e.g., "Box1"

        if len(line_lengths) >= 2:
            distance_ratio = line_lengths[0] / line_lengths[1]
        elif len(line_lengths) == 1:
            distance_ratio = 1
        else:
            distance_ratio = float('nan')

        if(distance_ratio != 1):
            all_ratios.append((box_number, distance_ratio))

        # print(all_ratios)

        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, image_with_lines)
        # print(f"✅ Saved with lines: {output_image_path}")

    # Save all ratios to CSV
    csv_filename = "workspace/distance_ratios.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Box Number", "Distance_Ratio"])
        writer.writerows(all_ratios)

    print(f"\n✅ All ratios saved to {csv_filename} and all lines saved to {output_folder}")

    ################################################ Predicting Age. ################################################

    # --- STEP 1: Load regression models for Box0, Box1, Box2 ---

    box_models = {}

    def load_model(box_csv, degree):
        df = pd.read_csv(box_csv)
        X = df[['Distance_Ratio']].values
        y = df['Age'].values

        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        return model, poly

    # Load models (adjust degrees if needed)
    box_models['Box0'] = load_model("Resources/Box0.csv", degree=3)
    box_models['Box1'] = load_model("Resources/Box1.csv", degree=3)
    box_models['Box2'] = load_model("Resources/Box2.csv", degree=3)

    # --- STEP 2: Process all CSV files in the folder ---
    file_path = "workspace/distance_ratios.csv"
    output_rows = []

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        predicted_ages = []

        for _, row in df.iterrows():
            box = row['Box Number']
            ratio = row['Distance_Ratio']

            if(ratio == 1):
                continue
            # ✅ Skip invalid boxes or NaN ratios
            if box in box_models and not pd.isna(ratio):
                model, poly = box_models[box]
                ratio_transformed = poly.transform(np.array([[ratio]]))
                predicted_age = model.predict(ratio_transformed)[0]
                predicted_ages.append(predicted_age)

        final_age = round(np.mean(predicted_ages), 3) if predicted_ages else None
        if(is_female and final_age != None):
            final_age = final_age-2
        if(final_age == None):
            final_age = -1
            
    print(f"Predicted age -> {final_age}")
    
    age_years = round(final_age, 2)   
    age_months = round(final_age * 12, 2)

    return age_years, age_months 

if __name__ == "__main__":
    ################################################ Input Image and male/female ################################################

    input_image_path = 'm16.png'
    is_female = False
    delete_old_files_and_folders()
    age = main_function(input_image_path, is_female)
    print(age)

