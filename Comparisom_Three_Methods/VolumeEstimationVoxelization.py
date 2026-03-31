# --------------------------------------------------------------------------------------
# Developer: Luciano Gonçalves Moreira
# Version: 1.0
# Date: March 20, 2026
# Institution: Universidade Federal de Viçosa (UFV) / 
#              Instituto Federal do Sudeste de Minas Gerais (IF Sudeste MG)
#
# Project: Two-Stage 3D Volume Estimation from LiDAR Point Clouds
# Module: LiDAR Volume Estimation (Voxelization)
#
# Description: This script accumulates real-time LiDAR point clouds and displays 
#              them in a 3D window using Open3D. It identifies, segments, and 
#              calculates the volume of a specific object using Voxelization. 
#              Finally, it displays the measurements, estimated volume, execution 
#              time, and saves the accumulated cloud to a PCD file.
# --------------------------------------------------------------------------------------

import time
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import scipy.spatial
from pathlib import Path    # Path from pathlib for file path manipulation
import csv                  # csv library for CSV file handling
import os                   # os library for file and directory manipulation

# Global variables for timing and trial control
start_time = 0              # Timer to measure execution time
test_count = 0              # Counter for LiDAR data collection attempts
successful_tests_count = 0  # Counter for successful tests

class LidarSubscriber(Node):
    """
    Main ROS node class that subscribes to the PointCloud2 message.
    """
    def __init__(self):
        super().__init__('lidar_voxel_node') # ROS node name
        
        # Subscriber setup (message type, topic, callback function, queue size)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/unilidar/cloud',
            self.callback,
            10
        )
        self.subscription  # Prevent unused variable warning

        # Defining 3D space boundaries for point filtering
        self.z_min = 0.5    # Minimum height
        self.z_max = 1.5    # Maximum height
        self.x_min = -0.5   # Minimum X-axis limit
        self.x_max = 1.5    # Maximum X-axis limit
        self.y_min = -1.5   # Minimum Y-axis limit
        self.y_max = 1.5    # Maximum Y-axis limit
        self.accumulated_points = []

        # Standard volume for comparison (adjustable as needed)
        self.standard_volume = 0.12200 # Reference volume in m³

        # Open3D Visualizer Initialization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Occupancy Grid 3D", width=800, height=600)
        self.point_cloud = o3d.geometry.PointCloud() # Empty point cloud object
        self.geometry_added = False                  # Flag to check if geometry is in the visualizer
        self.vis.get_render_option().point_size = 0.5 # Set point size for visualization

    def callback(self, msg):
        """
        Callback function triggered when a new PointCloud2 message is received.
        """
        # Convert PointCloud2 message to NumPy array
        new_points = self.convert_pointcloud2_to_numpy(msg)
        
        # Filter points based on the defined 3D boundaries
        mask = (
            (new_points[:, 0] >= self.x_min) & (new_points[:, 0] <= self.x_max) &
            (new_points[:, 1] >= self.y_min) & (new_points[:, 1] <= self.y_max) &
            (new_points[:, 2] >= self.z_min) & (new_points[:, 2] <= self.z_max)
        )
        filtered_points = new_points[mask]
        
        # Append new filtered points to the accumulated list
        self.accumulated_points.extend(filtered_points)

        # Update Open3D point cloud object
        self.point_cloud.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))
        
        # Add geometry if it's the first time, otherwise update it
        if not self.geometry_added:
            self.vis.add_geometry(self.point_cloud)
            self.geometry_added = True
        else:
            self.vis.update_geometry(self.point_cloud)
            
        self.vis.poll_events()      # Process visualizer events
        self.vis.update_renderer()  # Update visualizer renderer

    def convert_pointcloud2_to_numpy(self, msg):
        """
        Function to convert PointCloud2 message into a NumPy array.
        """
        point_step = msg.point_step  # Size of each point in the message
        data = msg.data              # Raw PointCloud2 message data
        points = []                  # List to store converted points
        
        for i in range(0, len(data), point_step):
            try:
                # Unpack binary data to obtain x, y, z coordinates
                x, y, z = struct.unpack_from('fff', data, i)
                points.append([x, y, z])
            except struct.error:
                # Skip points with incomplete data
                continue
        return np.array(points)
    
    def detect_and_segment_object(self):
        """
        Main function to detect and segment the object in the accumulated point cloud.
        """
        
        # ================ Internal Helper Function =========================
        def calculate_voxelization_volume(pcd):
            """
            Calculates volume and dimensions using the Voxel Grid method.
            """
            voxel_size = 0.01  # 1 cm voxel size
            sensor_ground_height = 1.55
            
            # Height filter to remove the floor
            z_values = np.asarray(pcd.points)[:, 2]
            object_top = np.percentile(z_values, 5) # 5th percentile for top surface
            object_height = sensor_ground_height - object_top - 0.10 # 10cm offset
            object_height = max(0.01, object_height)

            # Apply Voxelization
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

            # Get centers of occupied voxels
            occupied_voxels = voxel_grid.get_voxels()
            voxel_centers = np.array([voxel.grid_index for voxel in occupied_voxels])
            voxel_centers = voxel_centers * voxel_size  # Convert indices to real coordinates

            # Calculate Length and Width (XY plane)
            if len(voxel_centers) > 0:
                min_x, max_x = np.min(voxel_centers[:, 0]), np.max(voxel_centers[:, 0])
                min_y, max_y = np.min(voxel_centers[:, 1]), np.max(voxel_centers[:, 1])
                width_obj = max_x - min_x
                length_obj = max_y - min_y
            else:
                width_obj = 0.0
                length_obj = 0.0
                
            if width_obj > length_obj:
                width_obj, length_obj = length_obj, width_obj

            # Estimate volume: Occupied voxels * voxel area * calculated height
            num_occupied_voxels = len(occupied_voxels)
            estimated_volume = num_occupied_voxels * (voxel_size ** 2) * object_height 

            print(f"\nDimensions (Voxelization):")
            print(f"Length: {length_obj:.3f} m | Width: {width_obj:.3f} m | Height: {object_height:.3f} m")
            print(f"Occupied Voxels: {num_occupied_voxels} | Voxel Size: {voxel_size:.3f} m")
            print(f"Estimated Volume: {estimated_volume:.6f} m³")

            return voxel_grid, estimated_volume, width_obj, length_obj, object_height
        # ===================================================================

        global successful_tests_count, start_time
        print("Starting full object detection with Voxelization...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))

        # Filter: Voxel Grid Downsampling
        pcd = pcd.voxel_down_sample(voxel_size=0.015)

        # Filter: Statistical Outlier Removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)

        # Execute RANSAC Plane Segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.008,
                                                ransac_n=3,
                                                num_iterations=2000)
        # Keep only inlier points (well-fitted to the plane)
        pcd = pcd.select_by_index(inliers)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

        # DBSCAN Clustering for segmentation
        labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
        max_label = labels.max()
        print(f"{max_label + 1} clusters found.")

        if max_label < 0:
            print("No clusters found.")
            return

        clusters = []
        centers = []

        for i in range(max_label + 1):
            indices = np.where(labels == i)[0]
            cluster_points = np.asarray(pcd.points)[indices]
            if len(cluster_points) >= 100: # Minimum cluster size
                clusters.append(cluster_points)
                centers.append(np.mean(cluster_points, axis=0))

        if not clusters:
            print("No clusters of sufficient size found.")
            return

        # Identify the largest cluster
        cluster_sizes = [len(c) for c in clusters]
        largest_idx = np.argmax(cluster_sizes)
        largest_center = centers[largest_idx]

        # Merge clusters near the largest one
        merged_points = clusters[largest_idx]
        for i, cluster in enumerate(clusters):
            if i == largest_idx:
                continue
            dist = np.linalg.norm(np.array(centers[i]) - np.array(largest_center))
            if dist < 0.2:  # Proximity tolerance (adjustable)
                merged_points = np.vstack((merged_points, cluster))

        print(f"Object identified with {merged_points.shape[0]} points.")

        # Finalize execution time
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")
    
        # Create object point cloud from merged points
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(merged_points)
        
        # Calculate volume using voxelization helper
        voxel_grid, volume, width_obj, length_obj, height_obj = calculate_voxelization_volume(object_pcd)

        # Success check based on standard volume tolerance
        if ((volume >= (self.standard_volume - 0.005)) and (volume <= (self.standard_volume + 0.005))):
                print("Successful attempt! Volume is within standard tolerance.") 
                successful_tests_count += 1
                
                # Directory setup
                output_dir = Path.home() / "PCDs/Voxelization/OtherTests/Package_1_T1"
                output_dir.mkdir(parents=True, exist_ok=True)

                num = successful_tests_count
                pcd_path = output_dir / f"accumulated_cloud_{num}.pcd"
                csv_path = output_dir / f"volume_result_{num}.csv"

                # Save Point Cloud
                o3d.io.write_point_cloud(str(pcd_path), pcd)
                print(f"Accumulated point cloud saved to '{pcd_path}'.")

                # Prepare data for logging
                data_to_save = {
                    "Length (m)": round(length_obj, 3),
                    "Width (m)": round(width_obj, 3),
                    "Height (m)": round(height_obj, 3),
                    "Calculated Volume (m3)": round(volume, 6),
                    "Execution Time (s)": round(elapsed_time, 2)
                }

                # Save individual CSV for each trial
                with open(csv_path, mode='w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                    writer.writeheader()
                    writer.writerow(data_to_save)

                print(f"Results saved to: {csv_path}")

                # Visualization
                sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                o3d.visualization.draw_geometries(
                        [object_pcd, voxel_grid, sensor_frame],
                        window_name="Object with Voxelization",
                        width=800,
                        height=600
                    )


def main():
    global test_count, successful_tests_count, start_time
    acquisition_time = 20  # Data collection duration in seconds
    
    print("Starting LiDAR data collection...")
    
    # Trial loop
    while test_count < 10:  # Try up to 10 attempts
        rclpy.init()
        node = LidarSubscriber()
        start_time = time.time() # Reset timer for the attempt
        
        while (time.time() - start_time) < acquisition_time:
            rclpy.spin_once(node, timeout_sec=0.1)
            # Detect and segment after/during collection as per source logic
            node.detect_and_segment_object()
            time.sleep(0.1) # Short pause to ensure processing is complete
            
        test_count += 1
        print(f"Attempt {test_count} completed.")
        node.destroy_node()
        rclpy.shutdown()
        
        time.sleep(1) # 1-second pause between attempts
        
        # Check if 3 successful attempts have been recorded
        if successful_tests_count == 3:
            print("3 successful attempts recorded. Shutting down the program.")
            break

if __name__ == '__main__':
    main()