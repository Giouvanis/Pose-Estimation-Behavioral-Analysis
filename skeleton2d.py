import os
import sys
import cv2
import math
import yaml
import json
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import functional as F
from torchvision.transforms import Pad
import hrnet

# -----------------------------
# Calibration loader
# -----------------------------
def load_calibration(calib_json_path, camera_index=0):
    """Load intrinsics from calibration JSON."""
    with open(calib_json_path, 'r') as f:
        calib = json.load(f)

    cam = calib["Calibration"]["cameras"][camera_index]
    params = cam["model"]["ptr_wrapper"]["data"]["parameters"]

    fx = params["f"]["val"]
    fy = params["f"]["val"]
    cx = params["cx"]["val"]
    cy = params["cy"]["val"]

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)

    dist = np.array([
        params["k1"]["val"],
        params["k2"]["val"],
        params["p1"]["val"],
        params["p2"]["val"],
        params["k3"]["val"]
    ], dtype=np.float32)

    return K, dist, fx, fy, cx, cy


# -----------------------------
# Preprocessing utils
# -----------------------------
def calculate_padding(original_width, original_height, target_width, target_height):
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height
    if original_aspect_ratio > target_aspect_ratio:
        new_height = original_width / target_aspect_ratio
        padding_height = (new_height - original_height) / 2
        padding_width = 0
    else:
        new_width = original_height * target_aspect_ratio
        padding_width = (new_width - original_width) / 2
        padding_height = 0
    return int(padding_width), int(padding_height)


def add_margin_bbox(bbox, margin=0.05):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_w = width * margin
    pad_h = height * margin
    return [x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h]


def transform_frame(frame, target_width, target_height, bbox):
    x1, y1, x2, y2 = bbox
    transformed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    transformed_image = torch.from_numpy(transformed_image).permute(2, 0, 1)
    transformed_image = F.crop(transformed_image, int(y1), int(x1), int(y2 - y1), int(x2 - x1))
    padding_width, padding_height = calculate_padding(
        transformed_image.shape[2], transformed_image.shape[1], target_width, target_height
    )
    transformed_image = Pad(padding=(padding_width, padding_height), fill=0)(transformed_image)
    transformed_image = F.resize(transformed_image, (target_width, target_height))
    transformed_image = F.normalize(transformed_image, mean=[0.5] * 3, std=[0.5] * 3)
    return transformed_image, padding_width, padding_height


def extract_keypoints_with_confidence(heatmaps):
    keypoints_with_confidence = []
    for heatmap in heatmaps:
        heatmap = heatmap.squeeze(0)
        max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
        y, x = divmod(max_idx.item(), heatmap.size(1))
        confidence = max_val.item()
        keypoints_with_confidence.append(((x, y), confidence))
    return keypoints_with_confidence


# -----------------------------
# Skeleton drawing
# -----------------------------
SKELETON_CONNECTIONS = [
    (6, 4), (6, 5),   # nose → front paws
    (0, 1), (2, 3),   # back paws → wrists
    (7, 9), (9, 10), (10, 11), (11, 8)  # tail chain
]

def draw_skeleton(frame, keypoints, connections, color=(0, 0, 255)):
    for (i, j) in connections:
        if (not math.isnan(keypoints[i][0])) and (not math.isnan(keypoints[j][0])):
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, color, 2)
    for (x, y) in keypoints:
        if not math.isnan(x) and not math.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)


# -----------------------------
# Main inference
# -----------------------------
def infer_2d_skeleton(video_path, labels, yolo_weights, hrnet_yaml, hrnet_pth, calib_json, camera_index=0,
                      n_joints=12, confidence_threshold=0.1, output_suffix="-2d-skeleton"):

    # Load calibration
    K, dist, fx, fy, cx, cy = load_calibration(calib_json, camera_index)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        sys.exit(1)

    # Prepare output video
    base, ext = os.path.splitext(video_path)
    output_video_path = f"{base}{output_suffix}{ext}"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Load HRNet
    with open(hrnet_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['NUM_JOINTS'] = n_joints
    input_size = cfg['MODEL']['IMAGE_SIZE']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pose = hrnet.get_pose_net(cfg, is_train=False).to(device)
    state_dict = torch.load(hrnet_pth, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "final_layer" not in k}
    model_pose.load_state_dict(filtered_state_dict, strict=False)
    model_pose.eval()

    # Load YOLO
    model_detect = YOLO(yolo_weights)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort
        frame = cv2.undistort(frame, K, dist)

        # Detect
        result = model_detect(frame, stream=False, verbose=False)[0]
        if len(result.boxes.xyxy) == 0:
            video_writer.write(frame)
            continue

        bbox = add_margin_bbox(result.boxes.xyxy[0].cpu().numpy(), margin=0.03)

        # HRNet transform
        transformed_image, pad_w, pad_h = transform_frame(frame, *input_size, bbox)
        transformed_image = transformed_image.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model_pose(transformed_image).squeeze(0)

        resized_heatmaps = torch.stack([
            F.resize(hm.unsqueeze(0), (input_size[1], input_size[0])) for hm in predictions
        ])
        keypoints_and_conf = extract_keypoints_with_confidence(resized_heatmaps)

        keypoints = np.array([
            k if c > confidence_threshold else (float('nan'), float('nan'))
            for k, c in keypoints_and_conf
        ], dtype=np.float32)

        # Map back
        scale_x = input_size[0] / (bbox[2] - bbox[0] + 2 * pad_w)
        scale_y = input_size[1] / (bbox[3] - bbox[1] + 2 * pad_h)
        keypoints_orig = keypoints / np.array([scale_x, scale_y]) + np.array([bbox[0], bbox[1]]) - np.array([pad_w, pad_h])

        # Draw skeleton
        draw_skeleton(frame, keypoints_orig, SKELETON_CONNECTIONS)

        video_writer.write(frame)
        frame_idx += 1
        print(f"Processed frame {frame_idx}", end="\r")

    cap.release()
    video_writer.release()
    print("\n✅ 2D skeleton video saved:", output_video_path)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    labels = {
        0: 'back_back_paw',
        1: 'back_left_wrist',
        2: 'back_right_paw',
        3: 'back_right_wrist',
        4: 'front_left_paw',
        5: 'front_right_paw',
        6: 'nose',
        7: 'tail_base',
        8: 'tail_end',
        9: 'tail_lower_midpoint',
        10: 'tail_midpoint',
        11: 'tail_upper_midpoint'
    }

    VIDEO_PATH = sys.argv[1]
    YOLO_WEIGHTS = sys.argv[2]
    HRNET_YAML = sys.argv[3]
    HRNET_PTH = sys.argv[4]
    CALIB_JSON = sys.argv[5]

    infer_2d_skeleton(
        VIDEO_PATH,
        labels=labels,
        yolo_weights=YOLO_WEIGHTS,
        hrnet_yaml=HRNET_YAML,
        hrnet_pth=HRNET_PTH,
        calib_json=CALIB_JSON,
        camera_index=0,
        n_joints=12
    )
