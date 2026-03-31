# --------------------------------------------------------------------------------------
# Developer: Luciano Gonçalves Moreira
# Version: 1.0
# Date: March 20, 2026
# Institution: Universidade Federal de Viçosa (UFV) / 
#              Instituto Federal do Sudeste de Minas Gerais (IF Sudeste MG)
#
# Project: Two-Stage 3D Volume Estimation from LiDAR Point Clouds
# Module: LiDAR Volume Estimation (Bounding Box)
#
# Description: This script accumulates real-time LiDAR point clouds and displays 
#              them in a 3D window using Open3D. It identifies, segments, and 
#              calculates the volume of a specific object using bounding boxes. 
#              Finally, it displays the measurements, estimated volume, execution 
#              time, and saves the accumulated cloud to a PCD file.
# --------------------------------------------------------------------------------------

import time                     # Time library to measure execution time
import numpy as np              # NumPy library for array manipulation
import open3d as o3d            # Open3D library for point cloud visualization and manipulation
import rclpy                    # rclpy library for ROS 2 integration
from rclpy.node import Node     # Node class from rclpy to create a ROS node
from sensor_msgs.msg import PointCloud2  # ROS PointCloud2 message for handling point clouds
import struct                   # struct library for unpacking binary data
from pathlib import Path        # Path from pathlib for file path manipulation
import csv                      # csv library for CSV file handling
import os                       # os library for file and directory manipulation

# Global variables for timing and trial control
start_time = 0                  # Timer to measure execution time
test_count = 0                  # Counter for LiDAR data collection attempts
successful_tests_count = 0      # Counter for successful tests

class LidarSubscriber(Node):
    """
    Main ROS node class that subscribes to the PointCloud2 message.
    """
    def __init__(self):
        super().__init__('lidar_volume_node')  # ROS node name
        
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

        self.accumulated_points = []    # List to store accumulated points from LiDAR

        # Standard volume for comparison (adjustable as needed)
        self.standard_volume = 0.064520 # Reference volume in m³

        # Open3D Visualizer Initialization
        self.vis = o3d.visualization.Visualizer() 
        self.vis.create_window("Occupancy Grid 3D", width=800, height=600)
        
        self.point_cloud = o3d.geometry.PointCloud()    # Empty point cloud object
        self.geometry_added = False                     # Flag to check if geometry is in the visualizer
        self.vis.get_render_option().point_size = 0.5   # Set point size for visualization

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
        
    # Function to convert PointCloud2 message into a NumPy array
    def convert_pointcloud2_to_numpy(self, msg):
        point_step = msg.point_step  # Size of each point in the message
        data = msg.data              # Raw PointCloud2 message data
        points = []                  # List to store converted points
        
        for i in range(0, len(data), point_step):
            try:
                # Unpack binary data to obtain x, y, z coordinates
                x, y, z = struct.unpack_from('fff', data, i)
                points.append([x, y, z])
            except struct.error:
                # Skip points with incomplete data and restart loop
                continue
        return np.array(points)

    # Function to detect and segment the object in the accumulated point cloud
    def detect_and_segment_object(self):
        # ================ Helper Functions =========================
        
        # Function to create a solid OBB and its edges for visualization
        def create_solid_obb(obb, color=(0, 0, 1)):
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb)
            mesh.compute_vertex_normals()
            
            # Edges for contour visualization
            edges = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            edges.paint_uniform_color([1, 0, 0])  # Red color for edges
            
            return mesh, edges

        # Function to calculate OBB dimensions and volume
        def calculate_dimensions_and_volume(obb):
            volume = 0
            length = 0
            width = 0
            height = 0
            sensor_ground_height = 1.55  # Sensor height relative to the ground

            # --- Object Height Calculation ---
            z_values = np.asarray(pcd.points)[:, 2]     # Z values from point cloud
            object_top = np.percentile(z_values, 5)     # 5th percentile for top surface (adjustable)
            
            # Height relative to the floor platform (e.g., 10cm/0.10 offset)
            object_height = sensor_ground_height - object_top - 0.10
            object_height = max(0.01, object_height)    # Ensure minimum height

            # --- OBB Z-axis Identification ---
            z_global = np.array([0, 0, 1])              # Global Z vector pointing down
            dot_products = np.abs(np.dot(obb.R.T, z_global))
            z_index = np.argmax(dot_products)           # Index of the rotation vector closest to global Z

            # --- Dimension Calculation ---
            xy_indices = [i for i in range(3) if i != z_index]
            width = min(obb.extent[xy_indices[0]], obb.extent[xy_indices[1]])
            length = max(obb.extent[xy_indices[0]], obb.extent[xy_indices[1]])
            height = object_height

            # Volume calculation
            volume = width * length * height

            # Display results
            print(f"OBB Estimated Volume (raw): {obb.volume():.3f} m³")
            print("\nIdentified Object Dimensions:")
            print(f" Length: {length:.3f} m")
            print(f" Width:  {width:.3f} m")
            print(f" Height: {height:.3f} m")
            print(f" Volume: {volume:.6f} m³\n")

            # --- Downward Extrusion ---
            new_extent = np.array([length, width, object_height])
            new_center = obb.center.copy()
            new_center += obb.R[:, 2] * (object_height / 2)  

            corrected_obb = o3d.geometry.OrientedBoundingBox(new_center, obb.R, new_extent)
            return corrected_obb, volume, length, width, height

        # Function to create reference axes for the bounding box
        def create_axis_lines(origin, rotation, size=0.2):
            """
            Creates colored vectors representing local X, Y, Z axes of the OBB.
            """
            lines = []
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB for X, Y, Z

            line_points = []
            line_indices = []

            for i in range(3):
                start = origin
                end = origin + rotation[:, i] * size
                
                idx = len(line_points)
                line_points.append(start)
                line_points.append(end)
                line_indices.append([idx, idx + 1])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            return line_set

        # =================== End of Helper Functions =========================

        global successful_tests_count, start_time
        volume, length, width, height = 0, 0, 0, 0
        
        print("Starting object detection with bounding box...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))

        # Filter: Voxel Grid Downsampling
        voxel_size = 0.015  # Density between 0.005 and 0.02 recommended
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

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
            if len(cluster_points) >= 100:  # Minimum cluster size
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

        # Create object point cloud and OBB
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(merged_points)

        obb = object_pcd.get_minimal_oriented_bounding_box()
        obb.color = (1, 0, 0)
        obb_axes = create_axis_lines(origin=obb.center, rotation=obb.R, size=0.2)

        # Coordinate frame for the sensor
        sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

        # Calculate volume and Corrected OBB
        corrected_obb, volume, length, width, height = calculate_dimensions_and_volume(obb)

        # Finalize execution time
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")

        # Success check based on standard volume tolerance
        if ((volume >= (self.standard_volume - 0.002)) and (volume <= (self.standard_volume + 0.002))):
            print("Success! Volume is within standard tolerance.") 
            successful_tests_count += 1

            # Directory setup
            output_dir = Path.home() / "PCDs/BoundingBoxVolume/Package_7_T1"
            output_dir.mkdir(parents=True, exist_ok=True)

            num = successful_tests_count
            pcd_path = output_dir / f"accumulated_cloud_{num}.pcd"
            csv_path = output_dir / f"volume_result_{num}.csv"

            # Save Point Cloud and OBB Mesh
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            print(f"Accumulated point cloud saved to '{pcd_path}'.")

            # Log data
            data_to_save = {
                "OBB Volume (m3)": round(obb.volume(), 3),
                "Length (m)": round(length, 3),
                "Width (m)": round(width, 3),
                "Height (m)": round(height, 3),
                "Calculated Volume (m3)": round(volume, 6),
                "Execution Time (s)": round(elapsed_time, 2)
            }

            # Save individual CSV
            with open(csv_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                writer.writeheader()
                writer.writerow(data_to_save)

            print(f"Results saved to: {csv_path}")

            # --- Temporary Visualization ---
            o3d.visualization.draw_geometries(
                [pcd, corrected_obb, obb_axes, sensor_frame],
                window_name="OBB with Corrected Extrusion"
            )
            
def main():
    global test_count, successful_tests_count, start_time  # Reference global variables
    
    # Acquisition time: 0.25s (more precise), 0.3s (default), 90s (captures the full box)
    acquisition_time = 3 
    
    # Start of LiDAR data collection attempts
    print("Starting LiDAR data collection...")
    
    # Data collection loop
    while test_count < 10:  # Try up to 10 total attempts
        rclpy.init()
        node = LidarSubscriber()
        
        start_time = time.time()    # Reset the timer for the next attempt
        
        while (time.time() - start_time) < acquisition_time:
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # After the collection time, detect and segment the object
            node.detect_and_segment_object()
            time.sleep(0.1) # Short pause to ensure processing is complete
        
        test_count += 1
        print(f"Attempt {test_count} completed.")
        
        node.destroy_node()
        rclpy.shutdown()
        
        time.sleep(1)  # 1-second pause between attempts
        
        # Check if 3 successful attempts have been recorded        
        if successful_tests_count == 3:
            print("3 successful attempts recorded. Shutting down the program.")
            break

if __name__ == '__main__':
    main()
