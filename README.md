# Pose-Estimation-Behavioral-Analysis
This repository features a Python-based pipeline for visualizing and analyzing 3D skeletal data generated through pose estimation frameworks (such as Anipose). The project focuses on synchronizing triangulated 3D keypoints with raw video footage to validate behavioral events, specifically "standing" behaviors in animal models.
Key Features
3D Skeleton Reconstruction: Processes .pkl data to reconstruct a 35-keypoint anatomical skeleton, including facial features, spine, and limbs.  

🚀 Video Synchronization: Real-time side-by-side rendering of original MP4 video and the corresponding 3D matplotlib animation.  

Automated Behavior Detection: Logic-based event counter that identifies "standing" instances by comparing the vertical (Z-axis) coordinates of the nose, withers, and paws.  

Kinematic Visualization: Dynamic labeling of keypoints and skeleton chains for movement clarity.  

🛠 Tech Stack
Languages: Python

Libraries: OpenCV (Video processing), Matplotlib (3D Rendering), NumPy (Data manipulation), Pickle (Data serialization).  

Tools: Anipose (for initial triangulation).  

📂 Project Structure
skeleton_sync_vis.py: The main processing script for data loading, behavioral logic, and video export.  

combined_skeleton_and_video.mp4: (Example output) The final visualization showing the raw video alongside the reconstructed 3D pose.
