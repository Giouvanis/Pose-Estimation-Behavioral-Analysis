import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =================== USER CONFIG =======================
pkl_path = r"C:\Users\giouv\Behaviours\Anipose_Final\Triangulated_Data\Arena_video-12-49-31\results_trngl_optimized_arena_video-12-49-31.pkl"
video_path = r"C:\Users\giouv\OneDrive\Desktop\20241126-150123-406668_1.mp4"
output_path = r"C:\Users\giouv\OneDrive\Desktop\combined_skeleton_and_video.mp4"
# =======================================================

# Load 3D skeleton data
with open(pkl_path, "rb") as f:
    data = pickle.load(f)
p3ds = data[0] if isinstance(data, tuple) else data
n_frames, n_keypoints, _ = p3ds.shape

# Body parts and skeleton chains
bodyparts = [
    "nose", "right_eye", "right_ear_base", "right_ear_tip",
    "left_eye", "left_ear_base", "left_ear_tip",
    "lower_jaw", "throat", "chest", "tail_base",
    "head_midpoint", "back_withers", "back_midpoint", "back_croup",
    "tail_upper_midpoint", "tail_midpoint", "tail_lower_midpoint", "tail_end",
    "front_right_shoulder", "front_right_elbow", "front_right_wrist", "front_right_paw",
    "front_left_shoulder", "front_left_elbow", "front_left_wrist", "front_left_paw",
    "back_right_hip", "back_right_knee", "back_right_wrist", "back_right_paw",
    "back_left_hip", "back_left_knee", "back_left_wrist", "back_left_paw"
]
schemes = [
    ["nose", "right_eye", "right_ear_base", "right_ear_tip"],
    ["nose", "left_eye", "left_ear_base", "left_ear_tip"],
    ["nose", "lower_jaw", "throat", "chest", "tail_base"],
    ["nose", "head_midpoint", "back_withers", "back_midpoint", "back_croup", "tail_base",
     "tail_upper_midpoint", "tail_midpoint", "tail_lower_midpoint", "tail_end"],
    ["back_withers", "front_right_shoulder", "front_right_elbow", "front_right_wrist", "front_right_paw"],
    ["back_withers", "front_left_shoulder", "front_left_elbow", "front_left_wrist", "front_left_paw"],
    ["back_croup", "back_right_hip", "back_right_knee", "back_right_wrist", "back_right_paw"],
    ["back_croup", "back_left_hip", "back_left_knee", "back_left_wrist", "back_left_paw"]
]
skeleton_pairs = [(bodyparts.index(a), bodyparts.index(b)) for chain in schemes for a, b in zip(chain[:-1], chain[1:])]

# Key indices for standing check
nose_idx = bodyparts.index("nose")
withers_idx = bodyparts.index("back_withers")
fr_paw_r_idx = bodyparts.index("front_right_paw")
fr_paw_l_idx = bodyparts.index("front_left_paw")

# Video input
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up matplotlib 3D plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.nanmin(p3ds[:,:,0]), np.nanmax(p3ds[:,:,0]))
ax.set_ylim(np.nanmin(p3ds[:,:,1]), np.nanmax(p3ds[:,:,1]))
ax.set_zlim(np.nanmin(p3ds[:,:,2]), np.nanmax(p3ds[:,:,2]))
ax.view_init(elev=-115, azim=-90)
plt.tight_layout()

lines = [ax.plot([], [], [], c='gray')[0] for _ in skeleton_pairs]
label_texts = [ax.text(0, 0, 0, "", fontsize=7) for _ in range(n_keypoints)]
status_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=10)

# Get canvas size
fig.canvas.draw()
skel_img_w, skel_img_h = fig.canvas.get_width_height()

# Video writer
combined_w = vid_w + skel_img_w
combined_h = max(vid_h, skel_img_h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

# Standing counter
standing_counter = 0
was_standing = False

# Frame loop
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= n_frames:
        break

    pts = p3ds[frame_idx]

    # Plot skeleton lines
    for i, (a, b) in enumerate(skeleton_pairs):
        if not (np.any(np.isnan(pts[a])) or np.any(np.isnan(pts[b]))):
            xa, ya, za = pts[a]
            xb, yb, zb = pts[b]
            lines[i].set_data([xa, xb], [ya, yb])
            lines[i].set_3d_properties([za, zb])
        else:
            lines[i].set_data([], [])
            lines[i].set_3d_properties([])

    # Plot keypoint labels just below each keypoint
    for i, kp in enumerate(pts):
        if not np.any(np.isnan(kp)):
            label_texts[i].set_position((kp[0], kp[1]))
            label_texts[i].set_3d_properties(kp[2] - 5)  # label just below
            label_texts[i].set_text(bodyparts[i])
        else:
            label_texts[i].set_text("")

    # Standing condition check
    nose_z = pts[nose_idx, 2]
    withers_z = pts[withers_idx, 2]
    paw_r_z = pts[fr_paw_r_idx, 2]
    paw_l_z = pts[fr_paw_l_idx, 2]

    is_standing = nose_z > withers_z and paw_r_z > withers_z and paw_l_z > withers_z

    if is_standing and not was_standing:
        standing_counter += 1
    was_standing = is_standing

    status = f"Standing: {'YES' if is_standing else 'NO'} | Events: {standing_counter}"
    status_text.set_text(status)

    # Draw canvas
    ax.set_title(f"Frame {frame_idx+1}/{n_frames}")
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(skel_img_h, skel_img_w, 3)

    # Pad to match video height
    pad_video = cv2.copyMakeBorder(frame, 0, combined_h - vid_h, 0, 0, cv2.BORDER_CONSTANT)
    pad_plot = cv2.copyMakeBorder(plot_img, 0, combined_h - skel_img_h, 0, 0, cv2.BORDER_CONSTANT)
    combined = np.hstack((pad_video, pad_plot))

    out.write(combined)
    frame_idx += 1

# Finalize
cap.release()
out.release()
print(f"✅ Video saved to:\n{output_path}")
