# --------------------------------------------------------------------------------------
# Developer: Luciano Gonçalves Moreira
# Version: 1.0
# Date: March 20, 2026
# Institution: Universidade Federal de Viçosa (UFV) / 
#              Instituto Federal do Sudeste de Minas Gerais (IF Sudeste MG)
#
# Project: Two-Stage 3D Volume Estimation from LiDAR Point Clouds
# Module: LiDAR Volume Estimation (Convex Hull)
#
# Description: This script accumulates real-time LiDAR point clouds and displays them 
#              in a 3D window using Open3D. It identifies, segments, and calculates 
#              the volume of a specific object using a custom Convex Hull approach.
#              Finally, it saves the results (PCD, PLY, CSV) and displays measurements.
# --------------------------------------------------------------------------------------

import time
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
from sklearn.neighbors import KDTree
import scipy.spatial
from pathlib import Path
import csv
import os

# Timer for measuring execution time
start_time = 0   
# Global variables for trial and success control
test_count = 0 
successful_tests = 0 
volume_history = []  # List to store historical volumes for stabilization
end_sensor_read = False  # Flag to indicate if sensor reading is finished

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_volume_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/unilidar/cloud',
            self.callback,
            10
        )
        self.subscription

        # Defining 3D space boundaries (Cropping)
        self.z_min = 0.1
        self.z_max = 1.5
        self.x_min = -0.5
        self.x_max = 2.5
        self.y_min = -1.5
        self.y_max = 1.5
        self.accumulated_points = []

        # Reference volume for comparison (adjustable)
        self.standard_volume = 0.064500 # Standard volume in m³

        # Open3D Visualizer Initialization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Occupancy Grid 3D", width=800, height=600)
        self.point_cloud = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.vis.get_render_option().point_size = 0.5 

    def callback(self, msg):
        new_points = self.convert_pointcloud2_to_numpy(msg)
        
        # Apply spatial filter
        mask = (
            (new_points[:, 0] >= self.x_min) & (new_points[:, 0] <= self.x_max) &
            (new_points[:, 1] >= self.y_min) & (new_points[:, 1] <= self.y_max) &
            (new_points[:, 2] >= self.z_min) & (new_points[:, 2] <= self.z_max)
        )
        filtered_points = new_points[mask]
        self.accumulated_points.extend(filtered_points)

        # Update visualizer
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
        # Helper function to calculate volume via Convex Hull
        def calculate_convex_hull_volume(pcd, sensor_ground_height=1.55):
            volume = 0.0
            length_obj = 0.0
            width_obj = 0.0
            height_obj = 0.0

            # Calculate object height
            z_values = np.asarray(pcd.points)[:, 2] 
            object_top = np.percentile(z_values, 5) # 5% closest points to sensor (top)
            height_obj = sensor_ground_height - object_top - 0.10 # 10cm ground tolerance
            height_obj = max(0.01, height_obj) 

            # 1. Create 2D Convex Hull (XY projection)
            xy_points = np.asarray(pcd.points)[:, :2] 
            hull_2d = scipy.spatial.ConvexHull(xy_points) 
                         
            # 2. Combine two sets of points (top and bottom) to form the object "box"
            lower_verts = np.hstack([xy_points, np.full((len(xy_points), 1), object_top)]) 
            upper_verts = np.hstack([xy_points, np.full((len(xy_points), 1), object_top + height_obj)]) 
            all_verts = np.vstack([lower_verts, upper_verts]) 
            
            # 3. Create triangular faces surrounding the object
            faces = [] 
            n = len(hull_2d.vertices) 

            # Connect base and top points for lateral triangular faces
            for i in range(n):
                j = (i + 1) % n 
                v0 = hull_2d.vertices[i] 
                v1 = hull_2d.vertices[j] 
                v2 = v0 + len(xy_points) 
                v3 = v1 + len(xy_points) 

                faces.append([v0, v1, v2])  # Lower lateral triangle
                faces.append([v1, v3, v2])  # Upper lateral triangle

            # Base faces (using centroid + triangles)
            base_center_index = len(all_verts) 
            base_center = np.mean(lower_verts[hull_2d.vertices], axis=0) 
            all_verts = np.vstack([all_verts, base_center]) 

            for i in range(n):
                j = (i + 1) % n 
                v0 = hull_2d.vertices[i] 
                v1 = hull_2d.vertices[j] 
                faces.append([v0, v1, base_center_index]) 

            # Top faces (using centroid)
            top_center_index = len(all_verts) 
            top_center = np.mean(upper_verts[hull_2d.vertices], axis=0) 
            all_verts = np.vstack([all_verts, top_center])  

            for i in range(n):
                j = (i + 1) % n 
                v0 = hull_2d.vertices[i] + len(xy_points) 
                v1 = hull_2d.vertices[j] + len(xy_points) 
                faces.append([v0, v1, top_center_index]) 

            # 4. Convert faces to NumPy array
            try:
                if any(len(face) != 3 for face in faces):
                    raise ValueError("All faces must have exactly 3 vertices")
                faces_array = np.array(faces, dtype=np.int32) 
            except Exception as e:
                print(f"Error converting faces: {e}")
                raise

            # 5. Create Convex Hull mesh for 3D visualization
            hull_mesh = o3d.geometry.TriangleMesh() 
            hull_mesh.vertices = o3d.utility.Vector3dVector(all_verts) 
            hull_mesh.triangles = o3d.utility.Vector3iVector(faces_array) 
            hull_mesh.compute_vertex_normals() 

            # 6. Calculate estimated volume (base area * height)
            base_area = hull_2d.volume  # .volume in 2D Hull is actually the Area
            volume = base_area * height_obj 

            # Dimensions calculation
            width_obj = np.max(xy_points[hull_2d.vertices, 0]) - np.min(xy_points[hull_2d.vertices, 0]) 
            length_obj = np.max(xy_points[hull_2d.vertices, 1]) - np.min(xy_points[hull_2d.vertices, 1]) 
            if width_obj > length_obj:
                width_obj, length_obj = length_obj, width_obj

            print(f"\nObject Dimensions:")
            print(f"Length: {length_obj:.3f} m | Width: {width_obj:.3f} m | Height: {height_obj:.3f} m")
            print(f"\nDimensions (Convex Hull):")
            print(f"Base Area: {base_area:.3f} m² | Height: {height_obj:.3f} m")
            print(f"Total Volume: {volume:.6f} m³")

            return hull_mesh, volume, length_obj, width_obj, height_obj 

        # ---------------- Object detection and segmentation logic ----------------
        global successful_tests, start_time, volume_history, end_sensor_read
        volume, length_obj, width_obj, height_obj = 0.0, 0.0, 0.0, 0.0
        
        print("Starting full object detection with Convex Hull...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))

        # Filter: Voxel Grid Downsampling
        voxel_size = 0.015 
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

        # DBSCAN for cluster segmentation
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
            if len(cluster_points) >= 100: 
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
            if i == largest_idx: continue
            dist = np.linalg.norm(np.array(centers[i]) - np.array(largest_center))
            if dist < 0.2:  # Proximity tolerance
                merged_points = np.vstack((merged_points, cluster))

        print(f"Object identified with {merged_points.shape[0]} points.")

        # Execution time measurement
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")
 
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(merged_points)
        
        # Calculate Convex Hull and Volume
        hull_mesh, volume, length_obj, width_obj, height_obj = calculate_convex_hull_volume(object_pcd)
        hull_mesh.paint_uniform_color([0, 0.5, 0.8])  # Teal color
       
        # Check volume against standard tolerance
        if ((volume >= (self.standard_volume - 0.005)) and (volume <= (self.standard_volume + 0.005))):
            print("Successful attempt! Volume within standard range.") 
            successful_tests += 1 
            
            # Directory Setup
            output_dir = Path.home() / "PCDs/ConvexHullVolume/OtherTests/Test_1_P1"
            output_dir.mkdir(parents=True, exist_ok=True)

            num = successful_tests
            pcd_path = output_dir / f"accumulated_cloud_{num}.pcd"
            hull_mesh_path = output_dir / f"hull_mesh_{num}.ply"
            csv_path = output_dir / f"volume_result_{num}.csv"

            # Save Files
            o3d.io.write_triangle_mesh(str(hull_mesh_path), hull_mesh)
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            print(f"Accumulated point cloud saved to '{pcd_path}'.")

            # Data to save in CSV
            data_to_save = {
                "Length (m)": round(length_obj, 3),
                "Width (m)": round(width_obj, 3),
                "Height (m)": round(height_obj, 3),
                "Calculated Volume (m3)": round(volume, 6),
                "Execution Time (s)": round(elapsed_time, 2)
            }

            with open(csv_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                writer.writeheader()
                writer.writerow(data_to_save)

            print(f"Results saved to: {csv_path}")

            # Visualization
            sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries(
                [object_pcd, hull_mesh, sensor_frame],
                window_name="Object with Convex Hull",
                width=800, height=600
            )

def main():
    global test_count, successful_tests, start_time, end_sensor_read
    max_acquisition_time = 5 
    
    print("Starting LiDAR data collection...")
    
    while test_count < 10: 
        rclpy.init()
        node = LidarSubscriber()
        start_time = time.time()
        
        while ((time.time() - start_time) < max_acquisition_time) and not end_sensor_read:
            rclpy.spin_once(node, timeout_sec=0.1) 
            # Note: Processing occurs during collection per original logic
            node.detect_and_segment_object()
            time.sleep(0.1) # Short pause to ensure processing is complete
            
        test_count += 1
        node.destroy_node()
        rclpy.shutdown()
        
        time.sleep(1) # Interval between attempts
        print(f"Attempt {test_count} completed.")
        end_sensor_read = False
        
        if successful_tests == 3:
            print("3 successful attempts recorded. Closing program.")
            break

if __name__ == '__main__':
    main()