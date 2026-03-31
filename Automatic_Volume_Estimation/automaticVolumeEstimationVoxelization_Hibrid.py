# --------------------------------------------------------------------------------------
# Developer: Luciano Gonçalves Moreira
# Version: 1.0
# Date: March 20, 2026
# Institution: Universidade Federal de Viçosa (UFV) / 
#              Instituto Federal do Sudeste de Minas Gerais (IF Sudeste MG)
#
# Project: Two-Stage 3D Volume Estimation from LiDAR Point Clouds
# Module: LiDAR Volume Estimation (Hybrid Voxelization with Stabilization)
#
# Description: This script accumulates real-time LiDAR point clouds and displays 
#              them in a 3D window using Open3D. It identifies, segments, and 
#              calculates the volume using a hybrid voxelization approach (XY and YZ 
#              projections). It waits for the volume reading to stabilize before 
#              saving the accumulated point cloud and appending results to a CSV file.
# --------------------------------------------------------------------------------------

import sys
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
from sklearn.cluster import DBSCAN  # Importing DBSCAN for point cloud segmentation

start_time = 0   # Timer to measure execution time

# Global variables for timing and trial control
test_count = 0              # Counter for LiDAR data collection attempts
successful_tests_count = 0  # Counter for successful tests
volume_history = []         # List to store calculated volumes for stabilization check
end_sensor_read = False     # Flag to control the end of sensor reading

start_time = time.time()

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_hybrid_voxel_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/unilidar/cloud',
            self.callback,
            10
        )
        self.subscription

        # Defining 3D space boundaries for point filtering
        self.z_min = 0.15
        self.z_max = 1.5
        self.x_min = -0.25
        self.x_max = 2.0
        self.y_min = -1.5
        self.y_max = 1.5
        self.accumulated_points = []

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(f"Point Cloud Attempt {test_count}", width=800, height=600)
        self.point_cloud = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.vis.get_render_option().point_size = 0.5 # Set point size for visualization

    def callback(self, msg):
        new_points = self.convert_pointcloud2_to_numpy(msg)
        new_points = new_points[(new_points[:, 0] >= self.x_min) & (new_points[:, 0] <= self.x_max) &
                                (new_points[:, 1] >= self.y_min) & (new_points[:, 1] <= self.y_max) &
                                (new_points[:, 2] >= self.z_min) & (new_points[:, 2] <= self.z_max)]
        self.accumulated_points.extend(new_points)

        self.point_cloud.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))
        if not self.geometry_added:
            self.vis.add_geometry(self.point_cloud)
            self.geometry_added = True
        else:
            self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def convert_pointcloud2_to_numpy(self, msg):
        point_step = msg.point_step
        data = msg.data
        points = []
        for i in range(0, len(data), point_step):
            try:
                x, y, z = struct.unpack_from('fff', data, i)
                points.append([x, y, z])
            except struct.error:
                continue
        return np.array(points)
    
    def detect_and_segment_object(self):       
        # Function to check if volume has stabilized
        def has_volume_stabilized(current_volume, vol_history):
            """
            Checks if the estimated volume has stabilized based on the variation of the last readings.

            Parameters:
            - current_volume: float - volume calculated in the current iteration.
            - vol_history: list[float] - list with the last recorded volumes.
            
            Returns:
            - (bool): True if the volume has stabilized, False otherwise.
            """
            window_size = 3  # Number of consecutive volumes to consider
            delta_thresh = 0.00025  # Faster 0.00025 for rectangular boxes, more precise 0.000025
            
            # Add current volume to history
            vol_history.append(current_volume)

            # Keep history limited to the window size
            if len(vol_history) > window_size:
                vol_history.pop(0)

            # If we don't have enough history yet, it hasn't stabilized
            if len(vol_history) < window_size:
                return False

            # Check variations between consecutive volumes
            variations = [abs(vol_history[i] - vol_history[i - 1]) for i in range(1, window_size)]
            
            # If all variations are within the threshold, we consider it stabilized
            if all(v < delta_thresh for v in variations):
                return True

            return False

        # Function to calculate volume via Hybrid Voxelization
        def calculate_hybrid_volume(pcd, voxel_size=0.03, eps_z=0.03, eps_x=0.03, min_voxels=30):
            """
            Estimates the volume of irregular objects by combining XY (top) and YZ (face) projections,
            and averaging the estimated volumes to improve accuracy.

            Parameters:
            - pcd: open3d.geometry.PointCloud - segmented point cloud.
            - voxel_size: float - voxel size. (default 0.03)
            - eps_z: float - top clustering parameter on Z axis. (default 0.03)
            - eps_x: float - face clustering parameter on X axis. (default 0.03)
            - min_voxels: int - minimum voxels per group. (default 30)

            Returns:
            - total_volume: float - average estimated volume.
            - voxel_grid: open3d.geometry.VoxelGrid - generated voxel grid.
            """

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            voxels = voxel_grid.get_voxels()
            if len(voxels) == 0:
                print("No voxels found.")
                return voxel_grid, 0.0

            voxel_centers_idx = np.array([v.grid_index for v in voxels])
            voxel_centers_real = voxel_centers_idx * voxel_size

            z_floor = np.max(voxel_centers_real[:, 2])  # Highest Z (closest to the floor)
            x_depth = np.max(voxel_centers_real[:, 0])  # Highest X (depth/background)

            top_volume = 0.0
            face_volume = 0.0

            # --- TOP: XY projection + relative height ---
            z_values = voxel_centers_real[:, 2].reshape(-1, 1) 
            db_top = DBSCAN(eps=eps_z, min_samples=min_voxels).fit(z_values)
            for label in set(db_top.labels_):
                if label == -1:
                    continue
                cluster = voxel_centers_real[db_top.labels_ == label]
                xy_cells = set((int(v[0]/voxel_size), int(v[1]/voxel_size)) for v in cluster)
                area_xy = len(xy_cells) * voxel_size**2
                mean_z = np.mean(cluster[:, 2])
                height = z_floor - mean_z
                top_volume += area_xy * height

            # --- FACE: YZ projection + relative depth ---
            x_values = voxel_centers_real[:, 0].reshape(-1, 1)
            db_face = DBSCAN(eps=eps_x, min_samples=min_voxels).fit(x_values)
            for label in set(db_face.labels_):
                if label == -1:
                    continue
                cluster = voxel_centers_real[db_face.labels_ == label]
                yz_cells = set((int(v[1]/voxel_size), int(v[2]/voxel_size)) for v in cluster)
                area_yz = len(yz_cells) * voxel_size**2
                mean_x = np.mean(cluster[:, 0])
                depth = x_depth - mean_x
                face_volume += area_yz * depth

            # Average volume between the two estimates
            total_volume = (top_volume + face_volume) / 2

            print(f"[TOP] Estimated volume: {top_volume:.6f} m³")
            print(f"[FACE] Estimated volume: {face_volume:.6f} m³")
            print(f"[AVERAGE] Final estimated volume: {total_volume:.6f} m³")

            return voxel_grid, total_volume

        #------------------------------------------------------------------------------------
        # Start object detection and segmentation
        global successful_tests_count, start_time, volume_history, end_sensor_read 

        print("Starting full object detection with Hybrid Voxelization...")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))

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
            if len(cluster_points) >= 500: # Minimum cluster size
                clusters.append(cluster_points)
                centers.append(np.mean(cluster_points, axis=0))

        if not clusters:
            print("No clusters of sufficient size found.")
            return

        # Find the largest cluster
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
        
        # Calculate volume via Hybrid Voxelization
        voxel_grid, volume = calculate_hybrid_volume(object_pcd)

        if has_volume_stabilized(volume, volume_history):
            print("Volume has stabilized. Proceeding with calculation and saving.")
            end_sensor_read = True
            
        # Save accumulated point cloud and results
        if end_sensor_read:
            print("Successful attempt! Volume within standard range.")
            successful_tests_count += 1 
            
            # Directory Setup
            output_dir = Path.home() / "PCDs/Voxelization/RealTimeTests/LiveDemo" 
            output_dir.mkdir(parents=True, exist_ok=True)

            num = successful_tests_count
            pcd_path = output_dir / f"accumulated_cloud_{num}.pcd"
            csv_path = output_dir / f"volume_result.csv"

            # Save point cloud
            o3d.io.write_point_cloud(str(pcd_path), object_pcd)
            print(f"Accumulated point cloud saved to '{pcd_path}'.")

            # Data to save
            data_to_save = {
                "Attempt Number": num,
                "Number of Points": len(object_pcd.points),
                "Calculated Volume (m3)": round(volume, 6),
                "Execution Time (s)": round(elapsed_time, 2)
            }

            # Append to CSV for each successful attempt
            file_exists = os.path.exists(csv_path)

            with open(csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                
                if not file_exists:
                    writer.writeheader()  # Write header only the first time
                
                writer.writerow(data_to_save)
            print(f"Results saved to: {csv_path}")

            # Visualization
            sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries(
                [object_pcd, voxel_grid, sensor_frame],
                window_name=f"Object with Voxelization - Attempt {successful_tests_count}",
                width=800,
                height=600
            )


def main():
    global test_count, successful_tests_count, start_time, volume_history, end_sensor_read
    max_acquisition_time = 60  
    
    print("Starting LiDAR data collection...")
    
    # Trial loop
    while test_count < 10:  # Try up to 10 total attempts
        rclpy.init()
        node = LidarSubscriber()
        start_time = time.time()    # Reset timer for the attempt
        
        while ((time.time() - start_time) < max_acquisition_time) and not end_sensor_read:
            rclpy.spin_once(node, timeout_sec=0.1)
            # Detect and segment after/during collection
            node.detect_and_segment_object()
            
        test_count += 1
        print(f"Attempt {test_count} completed.")
        node.destroy_node()
        rclpy.shutdown()
        
        time.sleep(1)  # 1-second pause between attempts
        end_sensor_read = False
        
        # Check if 8 successful attempts have been recorded
        if successful_tests_count == 8:
            print("8 successful attempts recorded. Shutting down the program.")
            break

if __name__ == '__main__':
    main()