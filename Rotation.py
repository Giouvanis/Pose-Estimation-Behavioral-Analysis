import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with open("C:\Users\giouv\bnl-ai\bnl-ai-main\results_possible_optimized_arena_video-13-19-40.pkl", "rb") as f:
    points3d = pickle.load(f)  # Directly a NumPy array

marker_indices = [0, 1, 2]  # nose, back, tail

def compute_rotation_matrix_from_points(p1, p2, p3):
    x_axis = p2 - p1
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, p3 - p1)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return R

# === Preprocess: compute yaw angle and 2D positions for animation ===
rotations = []
yaws = []
positions = []

for i in range(points3d.shape[0]):
    pts = points3d[i]
    try:
        p1, p2, p3 = pts[marker_indices]
        if np.any(np.isnan([p1, p2, p3])):
            rotations.append(None)
            yaws.append(np.nan)
            positions.append(np.full((3, 2), np.nan))  # nose, back, tail in XY
            continue
        R = compute_rotation_matrix_from_points(p1, p2, p3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        yaws.append(yaw)
        positions.append(np.array([p1[:2], p2[:2], p3[:2]]))  # only XY
        rotations.append(R)
    except:
        rotations.append(None)
        yaws.append(np.nan)
        positions.append(np.full((3, 2), np.nan))

yaws = np.array(yaws)
positions = np.array(positions)
yaws_unwrapped = np.unwrap(yaws)

# === Plot & Animate ===
fig, ax = plt.subplots()
sc = ax.scatter([0, 0, 0], [0, 0, 0], c=["red", "green", "blue"], s=60)  # nose, back, tail
label_nose = ax.text(0, 0, "Nose", fontsize=10, color="red")
label_back = ax.text(0, 0, "Back", fontsize=10, color="green")
label_tail = ax.text(0, 0, "Tail", fontsize=10, color="blue")
text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')
ax.set_aspect("equal")


# Set axis limits based on all data
x_valid = positions[..., 0][~np.isnan(positions[..., 0])]
y_valid = positions[..., 1][~np.isnan(positions[..., 1])]
ax.set_xlim(np.min(x_valid), np.max(x_valid))
ax.set_ylim(np.min(y_valid), np.max(y_valid))
ax.set_title("2D Body Motion and Rotation Count")

def init():
    sc.set_offsets(np.zeros((3, 2)))
    text.set_text("")
    return sc, text

def update(frame):
    pts = positions[frame]
    if np.any(np.isnan(pts)):
        sc.set_offsets(np.full((3, 2), np.nan))
        text.set_text("Rotation count: N/A")
        label_nose.set_position((np.nan, np.nan))
        label_back.set_position((np.nan, np.nan))
        label_tail.set_position((np.nan, np.nan))
    else:
        sc.set_offsets(pts)
        # Compute rotation count up to this frame
        label_nose.set_position(pts[0])
        label_back.set_position(pts[1])
        label_tail.set_position(pts[2])
        delta_yaw = yaws_unwrapped[frame] - yaws_unwrapped[0]
        rot_count = delta_yaw / (2 * np.pi)
        text.set_text(f"Rotation count: {rot_count:.2f}")
    return sc, text, label_nose, label_back, label_tail

ani = FuncAnimation(fig, update, frames=len(positions), init_func=init,
                    interval=30, blit=True)

plt.show()
