import open3d as o3d
import numpy as np
import copy
import os


# Loading 3 point clouds

def load_point_clouds(voxel_size=0.0):
    pcds = []
    for i in range(1, 4):
        pcd = o3d.io.read_point_cloud("path_to_images%d.ply" %
                                      i)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds


voxel_size = 0.05  # Adjust this value based on the scale of point clouds
pcds_down = load_point_clouds(voxel_size)
o3d.visualization.draw_geometries(pcds_down,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])



# Perform pairwise ICP to infer the transformation difference

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print("Transformation is: ")
    print(icp_coarse.transformation)
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print("Transformation is: ")
    print(icp_fine.transformation)
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    print("Transformation is: ")
    print(information_icp)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph



print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

""" 
# Aligning pcd2 to pcd1
threshold = 0.02
trans_init = np.identity(4) # Identity matrix as initial guess

registration = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

# Applying the inferred transformation to align pcd2
pcd2.transform(registration.transformation)

# Repeat the process to align pcd3 to the now-combined cloud (pcd1 + pcd2)


#Define two 4x4 camera-to-world transformation matrices

T1 = np.eye(4) #Identity matrix representing the first camera pose (no transformation)
T2 = np.array([
    [0.98, 0.17, 0.0, 1.5],
    [-0.17, 0.98, 0.0, -0.5],
    [0.0, 0.0, 1.0, 0.2],
    [0.0, 0.0, 0.0, 1.0]
]) # Possible second camera pose with a slight rotation and translation

# Calculate the relative pose difference between the two camera positions
T_diff = np.linalg.inv(T1).dot(T2)
print("Relative Pose Transformation:\n", T_diff)

# Load your two point clouds
source = o3d.io.read_point_cloud("image1.ply")
target = o3d.io.read_point_cloud("image2.ply")

# Estimate normals for point-to-plane ICP (makes alignment much more precise)
source.estimate_normals()
target.estimate_normals()

# Define the maximum distance threshold for matching points
distance_threshold = 0.02 

# Run the ICP algorithm to get the transformation matrix
print("Applying ICP for alignment...")
icp_result = o3d.pipelines.registration.registration_icp(
    source, target, distance_threshold,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

# Apply the inferred transformation to align the source cloud with the target
source.transform(icp_result.transformation)
print("Transformation Matrix needed to align clouds:\n", icp_result.transformation)

# Visualize the result
o3d.visualization.draw_geometries([source, target])








def load_trajectory(filepath):

    traj = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Assuming space/tab-separated 16-value matrix rows, or 4x4 flattened array.
            values = [float(x) for x in line.strip().split()]
            if len(values) == 16:
                traj.append(np.array(values).reshape(4, 4))
    return traj

def save_trajectory(traj, filepath):

    with open(filepath, 'w') as f:
        for mat in traj:
            # Flatten 4x4 matrix and write space-separated on a single line
            flat_mat = mat.flatten()
            f.write(" ".join([f"{x:.6f}" for x in flat_mat]) + "\n")

def main():
   
    # Load the source files

    pcd1 = o3d.io.read_point_cloud("image1.ply")
    pcd2 = o3d.io.read_point_cloud("image2.ply")
    pcd3 = o3d.io.read_point_cloud("image3.ply")
    
    source_traj = load_trajectory("traj.txt")

    # Determine the rigid transformation required by comparing the source to your specific viewer's coordinate system. 

    theta = np.deg2rad(90)
    R_viewer = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta),  np.cos(theta)]
    ])
    T_viewer = np.array([0.0, 0.0, 0.0]) 
    
    # Construct the 4x4 transformation matrix

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_viewer
    transformation_matrix[:3, 3] = T_viewer

    # Alternatively, if you are aligning to another target point cloud from the viewer:
    # target_pcd = o3d.io.read_point_cloud("viewer_expected_cloud.ply")
    # reg_result = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, init_guess,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )
    # transformation_matrix = reg_result.transformation

    # Apply transformation to Point Clouds
    pcd1_transformed = copy.deepcopy(pcd1).transform(transformation_matrix)
    pcd2_transformed = copy.deepcopy(pcd2).transform(transformation_matrix)
    pcd3_transformed = copy.deepcopy(pcd3).transform(transformation_matrix)

    # Apply transformation to Trajectory Matrices

    transformed_traj = []
    for mat in source_traj:
        # T_new = T_alignment * T_source
        new_mat = np.dot(transformation_matrix, mat)
        transformed_traj.append(new_mat)


    # Save Replacement Files

    o3d.io.write_point_cloud("output_image1.ply", pcd1_transformed)
    o3d.io.write_point_cloud("output_image2.ply", pcd2_transformed)
    o3d.io.write_point_cloud("_output_image3.ply", pcd3_transformed)
    
    save_trajectory(transformed_traj, "output_traj.txt")
    print("Files successfully transformed and replaced!")

if __name__ == "__main__":
    main()

"""