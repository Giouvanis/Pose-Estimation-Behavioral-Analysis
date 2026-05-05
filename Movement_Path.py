import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load your triangulated data file
with open("C:\Users\giouv\bnl-ai\bnl-ai-mainresults_ransac_optimized_arena_video-13-19-40.pkl", "rb") as f:
    points3d = pickle.load(f)

marker_indices = [0, 1, 2]  # nose, back, tail

# Compute midpoint between nose and tail
trajectory_points = []
for i in range(points3d.shape[0]):
    try:
        nose, _, tail = points3d[i][marker_indices]
        if np.any(np.isnan([nose, tail])):
            trajectory_points.append([np.nan, np.nan])
            continue
        midpoint = (nose[:2] + tail[:2]) / 2  # 2D XY midpoint
        trajectory_points.append(midpoint)
    except:
        trajectory_points.append([np.nan, np.nan])

trajectory_points = np.array(trajectory_points)
valid_mask = ~np.isnan(trajectory_points).any(axis=1)
valid_traj = trajectory_points[valid_mask]

# Use center of motion as reference point
center = np.mean(valid_traj, axis=0)
rel_positions = valid_traj - center

# Compute angle from center
angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])
angles_unwrapped = np.unwrap(angles)
delta_angle = angles_unwrapped[-1] - angles_unwrapped[0]
num_loops = delta_angle / (2 * np.pi)

print(f"🐭 Estimated number of full loops: {num_loops:.2f}")

# Plot the movement
plt.figure(figsize=(6, 6))
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'o-', markersize=2, alpha=0.7)
plt.plot(center[0], center[1], 'rx', label='Center')
plt.title("Rat 2D Movement Path")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
