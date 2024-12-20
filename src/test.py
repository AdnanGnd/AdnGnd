from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
import numpy as np
import cv2

# Path to NuScenes mini dataset
dataset_path = '/mnt/d/nuscenes'
nusc = NuScenes(version='v1.0-mini', dataroot=dataset_path, verbose=True)

# Example: Accessing sample data
sample_token = nusc.sample[0]['token']
sample = nusc.get('sample', sample_token)

# Get camera data from right and left cameras (CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT)
front_cam_token = sample['data']['CAM_FRONT']
front_right_cam_token = sample['data']['CAM_FRONT_RIGHT']
front_left_cam_token = sample['data']['CAM_FRONT_LEFT']
front_cam_data = nusc.get('sample_data', front_cam_token)
front_right_cam_data = nusc.get('sample_data', front_right_cam_token)
front_left_cam_data = nusc.get('sample_data', front_left_cam_token)


# Load the images from the front-left and front-right cameras
front_image_path = nusc.get_sample_data_path(front_cam_data['token'])
left_image_path = nusc.get_sample_data_path(front_left_cam_data['token'])
right_image_path = nusc.get_sample_data_path(front_right_cam_data['token'])

front_image = cv2.imread(front_image_path)
left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)

# Get LiDAR/Radar data (LIDAR_TOP, RADAR_FRONT)
lidar_token = sample['data']['LIDAR_TOP']
radar_token = sample['data']['RADAR_FRONT']
lidar_data = nusc.get('sample_data', lidar_token)
radar_data = nusc.get('sample_data', radar_token)

lidar_path = nusc.get_sample_data_path(lidar_token)
lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

print(lidar_points.shape)


from PIL import Image

# Resize and normalize images using OpenCV
front_image = cv2.resize(front_image, (224, 224))  # Resize to 224x224 for CNN input
right_image = cv2.resize(right_image, (224, 224))
left_image = cv2.resize(left_image, (224, 224))

# Normalize pixel values to [0, 1]
front_image = front_image / 255.0
right_image = right_image / 255.0
left_image = left_image / 255.0




# # Example: Access LiDAR data and compute the distance of each point
# lidar_pc = nusc.get_sample_data(lidar_data['token'])['points']  # Get LiDAR points
# lidar_points = np.frombuffer(lidar_pc, dtype=np.float32).reshape((-1, 5))  # (x, y, z, intensity, timestamp)

# # Calculate the Euclidean distance of each LiDAR point from the sensor
# lidar_distance = np.linalg.norm(lidar_points[:, :3], axis=1)
# # You can now use lidar_distance for object detection and distance measurement




# import open3d as o3d

# # Convert LiDAR point cloud to Open3D format
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])

# # Visualize point cloud (optional)
# o3d.visualization.draw_geometries([pcd])

# # Convert LiDAR data to BEV (bird's-eye view)
# # You can project the 3D points onto a 2D plane (x, y), with the intensity as a color channel if needed
# bev_map = np.zeros((200, 200))  # Initialize an empty BEV map (adjust size as needed)

# # Project points onto 2D BEV
# for point in lidar_points:
#     x, y = int(point[0] * 10), int(point[1] * 10)  # Scale the x, y coordinates
#     if 0 <= x < 200 and 0 <= y < 200:
#         bev_map[x, y] = 1  # Mark the presence of a point in the BEV

# # Normalize the BEV map
# bev_map = bev_map / np.max(bev_map)



# # Example: Get annotations for a sample
# annotations = sample['anns']

# # Extract 3D bounding boxes and class labels
# boxes = []
# labels = []
# for ann_token in annotations:
#     ann = nusc.get('annotation', ann_token)
#     if ann['category'] == 'vehicle':  # Assuming we are detecting vehicles
#         box = Box(ann['translation'], ann['size'], ann['rotation'])
#         boxes.append(box)
#         labels.append(ann['category'])





# # Combine camera data and LiDAR distance
# inputs = {
#     'right_cam': right_cam_image,
#     'left_cam': left_cam_image,
#     'lidar_distance': lidar_distance  # You may want to concatenate this info with the image data
# }

# # Output annotations (bounding boxes and labels for vehicles)
# outputs = {
#     'boxes': boxes,
#     'labels': labels
# }
