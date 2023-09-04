'''
This script is written for finding the stable pose of the chair with its physical property in pybullet.
'''
from __future__ import print_function
import pybullet as p
import pybullet_data
import numpy as np
import math
import trimesh
import os

from utils import *

class ChairStabilityMatrix(object):
    """
    Use Exponential for checking rotation differences
    """
    def __init__(self, check_process=False, mp4_dir=None, angle_seg_num=15):
        ### Hyperparameter ###
        # Uniformly sample four angles in each direction from [0, 2*pi]
        self.angle_seg_num_x = angle_seg_num #15
        self.angle_seg_num_y = angle_seg_num #15

        print(f'[ChairStabilityMatrix] angle_seg_num: {self.angle_seg_num_x}')
        
        self.row_num = 5
        self.col_num = self.angle_seg_num_y / self.row_num

        assert self.angle_seg_num_y % self.row_num == 0

        # Distance between each chair
        self.obj_adj_dist = 2

        # Thresholds to determine if two poses are in the same functional pose set
        self.pos_threshold = 0.01
        self.orn_threshold = 0.01
        self.rot_diff_angle = 0.05
        self.z_axis_pos_threshold = 0.05  # position diff in z axis
        self.z_rot_axis_threshold = 0.05  # diff between rotation axis and z axis

        ### Simulation Parameter ###
        self.simulation_iteration = 1200
        self.start_calculation_iteration = self.simulation_iteration - 50
        self.check_process = check_process
        self.mp4_dir = mp4_dir

        self.obj_id_list = []

    def get_stable_pose_baseline(self, obj_urdf, transform_mesh):
        """Get the stable pose of the object by dropping with OBB faces
            parallel to the ground.

            Args:
                obj_urdf: urdf file
                transform_mesh: ob transformed mesh
            
            Returns:
                obj_stable_orn_list: list of stable orientation (quaternion)
                obj_stable_pos_list: list of stable position
                obj_stable_orn_eul_list: list of stable orientation (euler angle)
        """
        # Initialize the simulation
        if self.check_process:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF('plane.urdf')
        p.setGravity(0, 0, -9.81)

        # Bounding Box and COM height
        obj_mesh = trimesh.load(transform_mesh)
        obj_bbox = obj_mesh.extents
        obj_extent = math.sqrt(obj_bbox[0] * obj_bbox[0] +
                                obj_bbox[1] * obj_bbox[1] +
                                obj_bbox[2] * obj_bbox[2])
        obj_drop_height = obj_extent / 2 + 0.05

        # Drop initial position
        pos_list = []
        for i in range(6):
            pos = [0, i * self.obj_adj_dist, obj_drop_height]
            pos_list.append(pos)

        # Drop initial rotation
        axis = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]]
        quat_list = []
        for j in range(6):
            rotm = np.eye(3)
            rotm[2, :] = - np.array(axis[j])
            for i in range(3):
                temp_axis = np.zeros(3)
                temp_axis[i] = 1
                if abs(np.dot(rotm[2, :], temp_axis)) < 0.95:
                    y_axis = np.cross(rotm[2, :], temp_axis)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    rotm[1, :] = y_axis
                    x_axis = np.cross(y_axis, rotm[2, :])
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    rotm[0, :] = x_axis
                    break
            quat = rotm2quat(rotm)
            quat_list.append(quat)

        # Load chairs
        chair_id_list = []
        for i in range(6):
            chair_id = p.loadURDF(obj_urdf)
            p.resetBasePositionAndOrientation(chair_id, pos_list[i], quat_list[i])
            chair_id_list.append(chair_id)

        # List for saving stable orientation of the chair
        obj_stable_orn_list = []
        obj_stable_pos_list = []
        obj_stable_orn_matrix_list = []
        obj_stable_orn_eul_list = []

        for i in range(1200):
            p.stepSimulation()

        for i in range(6):
            obj_stable_pos_candidate, obj_stable_orn_candidate = p.getBasePositionAndOrientation(chair_id_list[i])
            obj_stable_orn_candidate_matrix = np.array(p.getMatrixFromQuaternion(obj_stable_orn_candidate)).reshape(3, 3)
            obj_stable_orn_candidate_eul = p.getEulerFromQuaternion(obj_stable_orn_candidate)

            # If the obj_stable_orn list is empty, insert the first stable orn and pos
            if not obj_stable_orn_list:
                obj_stable_orn_list.append(
                    obj_stable_orn_candidate)
                obj_stable_pos_list.append(
                    obj_stable_pos_candidate)
                obj_stable_orn_matrix_list.append(
                    obj_stable_orn_candidate_matrix)
                obj_stable_orn_eul_list.append(obj_stable_orn_candidate_eul)

            # Compare the current pose to those within the list
            else:
                insert_flag = True

                for stable_pose_idx, obj_stable_orn_member_matrix in enumerate(
                        obj_stable_orn_matrix_list):

                    #### Axis Angle ###
                    temp_rotm_diff = np.dot(
                        obj_stable_orn_member_matrix,
                        np.transpose(obj_stable_orn_candidate_matrix))

                    [temp_angle, _, _,
                        temp_z_axis] = rotm2angle(temp_rotm_diff)

                    temp_pos_diff = abs(
                        obj_stable_pos_candidate[-1] -
                        obj_stable_pos_list[stable_pose_idx][-1])

                    if temp_angle < self.rot_diff_angle:
                        insert_flag = False
                        break

                    if (1 - abs(temp_z_axis)
                        ) < self.z_rot_axis_threshold:
                        if temp_pos_diff < self.z_axis_pos_threshold:
                            insert_flag = False
                            break

                if insert_flag:
                    obj_stable_orn_list.append(
                        obj_stable_orn_candidate)
                    obj_stable_pos_list.append(
                        obj_stable_pos_candidate)
                    obj_stable_orn_matrix_list.append(
                        obj_stable_orn_candidate_matrix)
                    obj_stable_orn_eul_list.append(
                        obj_stable_orn_candidate_eul)

        p.disconnect()

        return obj_stable_orn_list, obj_stable_pos_list, obj_stable_orn_eul_list
        
    def get_stable_pose(self, obj_urdf, transform_mesh):
        """Get the stable pose of the object.
        
        Args:
            obj_urdf: urdf file
            transform_mesh: ob transformed mesh
            
        Returns:
            obj_stable_orn_list: list of stable orientation (quaternion)
            obj_stable_pos_list: list of stable position
            obj_stable_orn_eul_list: list of stable orientation (euler angle)
        """

        # Initialize the simulation
        if self.check_process:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the plane
        p.loadURDF("plane.urdf")

        # Bounding Box and COM height
        obj_mesh = trimesh.load(transform_mesh)
        obj_bbox = obj_mesh.extents
        obj_extent = math.sqrt(obj_bbox[0] * obj_bbox[0] +
                               obj_bbox[1] * obj_bbox[1] +
                               obj_bbox[2] * obj_bbox[2])
        obj_drop_height = obj_extent / 2 + 0.05

        p.resetDebugVisualizerCamera(obj_drop_height * 12, 90, -45, [
            self.obj_adj_dist *
            (self.col_num - 1) / 2 + 1 , self.obj_adj_dist * (self.row_num -1) / 2, 1
        ])
        ####################### Simulation ##########################
        # Save mp4 video
        if (self.mp4_dir is not None) and (self.check_process):
            obj_name = obj_urdf.split('/')[-1].split('.')[0]
            mp4_file_name = obj_name + "_stable_imagination.mp4"
            mp4_file_path = os.path.join(self.mp4_dir, mp4_file_name)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, mp4_file_path)

        # List for saving stable orientation of the chair
        obj_stable_orn_list = []
        obj_stable_pos_list = []
        obj_stable_orn_matrix_list = []
        obj_stable_orn_eul_list = []

        # Compute the accumulated pos and or change
        obj_pos_change = np.zeros(
            (self.angle_seg_num_x * self.angle_seg_num_y))
        obj_orn_change = np.zeros(
            (self.angle_seg_num_x * self.angle_seg_num_y))

        # Buffer for saving the prev pos and orn for computing the change
        prev_obj_pos = np.zeros(
            (3, self.angle_seg_num_x * self.angle_seg_num_y))
        prev_obj_orn = np.zeros(
            (4, self.angle_seg_num_x * self.angle_seg_num_y))

        # List to save the final obj pos and orn
        final_obj_pos = np.zeros(
            (3, self.angle_seg_num_x * self.angle_seg_num_y))
        final_obj_orn = np.zeros(
            (4, self.angle_seg_num_x * self.angle_seg_num_y))

        # Load the object into the simulator first
        for y_idx in range(self.angle_seg_num_y):
            obj_id = p.loadURDF(obj_urdf)
            p.changeDynamics(obj_id, -1, restitution=0.1)
            self.obj_id_list.append(obj_id)

        # Sampling the orientation about x and y axis
        for x_idx in range(self.angle_seg_num_x):
            for y_idx in range(self.angle_seg_num_y):
                # Drop orientation
                obj_start_orn_euler_x = 2 * np.pi * x_idx / self.angle_seg_num_x
                obj_start_orn_euler_y = 2 * np.pi * y_idx / self.angle_seg_num_y

                obj_start_orn_euler = [
                    obj_start_orn_euler_x,
                    obj_start_orn_euler_y,
                    0
                ]

                obj_start_orn = p.getQuaternionFromEuler(obj_start_orn_euler)

                col_idx = y_idx % self.row_num
                row_idx = (y_idx - col_idx) / self.row_num

                obj_start_x = row_idx * self.obj_adj_dist
                obj_start_y = col_idx * self.obj_adj_dist

                # Drop position
                obj_start_pos = [obj_start_x, obj_start_y, obj_drop_height]

                obj_id = self.obj_id_list[y_idx]

                p.resetBasePositionAndOrientation(obj_id, obj_start_pos,
                                                  obj_start_orn)

            # Start simulation
            for i in range(self.simulation_iteration):
                # Add random horizontal force
                x_random_force = 0.5 * np.random.random()
                y_random_force = 0.5 * np.random.random()
                p.setGravity(x_random_force, y_random_force, -10)

                p.stepSimulation()

                # Initialize prev_obj_pos and prev_obj_orn
                if i == self.start_calculation_iteration:
                    for y_idx in range(self.angle_seg_num_y):
                        obj_idx = x_idx * self.angle_seg_num_y + y_idx
                        obj_id = self.obj_id_list[y_idx]

                        temp_pos, temp_orn = p.getBasePositionAndOrientation(
                            obj_id)

                        prev_obj_pos[:, obj_idx] = np.array(temp_pos)
                        prev_obj_orn[:, obj_idx] = np.array(temp_orn)

                if i > self.start_calculation_iteration:
                    for y_idx in range(self.angle_seg_num_y):
                        obj_idx = x_idx * self.angle_seg_num_y + y_idx
                        obj_id = self.obj_id_list[y_idx]

                        temp_prev_obj_pos = prev_obj_pos[:, obj_idx]
                        temp_prev_obj_orn = prev_obj_orn[:, obj_idx]

                        temp_curr_obj_pos, temp_curr_obj_orn = p.getBasePositionAndOrientation(
                            obj_id)

                        temp_curr_obj_pos = np.array(temp_curr_obj_pos)
                        temp_curr_obj_orn = np.array(temp_curr_obj_orn)

                        ### Position change ###
                        obj_pos_change[obj_idx] += np.linalg.norm(
                            temp_prev_obj_pos - temp_curr_obj_pos)

                        ### Rotation change ###
                        temp_prev_obj_rotm = np.reshape(
                            np.array(
                                p.getMatrixFromQuaternion(temp_prev_obj_orn)),
                            (3, 3))
                        temp_curr_obj_rotm = np.reshape(
                            np.array(
                                p.getMatrixFromQuaternion(temp_curr_obj_orn)),
                            (3, 3))

                        obj_orn_relative_change = np.dot(
                            np.transpose(temp_prev_obj_rotm),
                            temp_curr_obj_rotm)
                        obj_orn_change[obj_idx] += matrixexponential(
                            obj_orn_relative_change)

                        # Update the prev_obj_pos and pre_obj_orn
                        prev_obj_pos[:, obj_idx] = temp_curr_obj_pos
                        prev_obj_orn[:, obj_idx] = temp_curr_obj_orn

                if i == (self.simulation_iteration - 1):
                    for y_idx in range(self.angle_seg_num_y):
                        obj_idx = x_idx * self.angle_seg_num_y + y_idx
                        obj_id = self.obj_id_list[y_idx]

                        temp_pos, temp_orn = p.getBasePositionAndOrientation(
                            obj_id)

                        final_obj_pos[:, obj_idx] = np.array(temp_pos)
                        final_obj_orn[:, obj_idx] = np.array(temp_orn)
            # import ipdb; ipdb.set_trace()
        p.disconnect()

        ########### Find stable pose ##########
        for x_idx in range(self.angle_seg_num_x):
            for y_idx in range(self.angle_seg_num_y):

                obj_idx = x_idx * self.angle_seg_num_y + y_idx

                temp_obj_pos_change = obj_pos_change[obj_idx]
                temp_obj_orn_change = obj_orn_change[obj_idx]

                # Check if the pose is stable
                if temp_obj_pos_change <= self.pos_threshold and temp_obj_orn_change <= self.orn_threshold:
                    obj_stable_pos_candidate = final_obj_pos[:, obj_idx]
                    obj_stable_orn_candidate = final_obj_orn[:, obj_idx]

                    # Calculate the rotational matrix about the x and y axis
                    obj_stable_orn_candidate_matrix = p.getMatrixFromQuaternion(
                        obj_stable_orn_candidate)
                    obj_stable_orn_candidate_matrix = np.array(
                        obj_stable_orn_candidate_matrix).reshape(3, 3)

                    obj_stable_orn_candidate_eul = p.getEulerFromQuaternion(obj_stable_orn_candidate)

                    # If the obj_stable_orn list is empty, insert the first stable orn and pos
                    if not obj_stable_orn_list:
                        obj_stable_orn_list.append(
                            np.ndarray.tolist(obj_stable_orn_candidate))
                        obj_stable_pos_list.append(
                            np.ndarray.tolist(obj_stable_pos_candidate))
                        obj_stable_orn_matrix_list.append(
                            obj_stable_orn_candidate_matrix)
                        obj_stable_orn_eul_list.append(obj_stable_orn_candidate_eul)

                    # Compare the current pose to those within the list
                    else:
                        insert_flag = True

                        for stable_pose_idx, obj_stable_orn_member_matrix in enumerate(
                                obj_stable_orn_matrix_list):

                            #### Axis Angle ###
                            temp_rotm_diff = np.dot(
                                obj_stable_orn_member_matrix,
                                np.transpose(obj_stable_orn_candidate_matrix))

                            [temp_angle, _, _,
                             temp_z_axis] = rotm2angle(temp_rotm_diff)

                            temp_pos_diff = abs(
                                obj_stable_pos_candidate[-1] -
                                obj_stable_pos_list[stable_pose_idx][-1])

                            if temp_angle < self.rot_diff_angle:
                                insert_flag = False
                                break

                            if (1 - abs(temp_z_axis)
                                ) < self.z_rot_axis_threshold:
                                if temp_pos_diff < self.z_axis_pos_threshold:
                                    insert_flag = False
                                    break

                        if insert_flag:
                            obj_stable_orn_list.append(
                                np.ndarray.tolist(obj_stable_orn_candidate))
                            obj_stable_pos_list.append(
                                np.ndarray.tolist(obj_stable_pos_candidate))
                            obj_stable_orn_matrix_list.append(
                                obj_stable_orn_candidate_matrix)
                            obj_stable_orn_eul_list.append(
                                obj_stable_orn_candidate_eul)

        print("Number of stable poses: %d" % len(obj_stable_orn_list))
        # self.visualize_result(obj_urdf, obj_stable_orn_list,
        #                  obj_stable_pos_list)
        return obj_stable_orn_list, obj_stable_pos_list, obj_stable_orn_eul_list

    def visualize_result(self, obj_urdf, obj_stable_orn_list,
                         obj_stable_pos_list):
        p.connect(p.GUI)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")
        # self.obj_adj_dist = 0.7
        p.resetDebugVisualizerCamera(3, 90, -45, [0, self.obj_adj_dist * 2, 0])

        for i in range(len(obj_stable_orn_list)):
            obj_orn = obj_stable_orn_list[i]
            obj_pos_x = 0
            obj_pos_y = self.obj_adj_dist * i
            obj_pos_z = obj_stable_pos_list[i][-1]
            obj_pos = (obj_pos_x, obj_pos_y, obj_pos_z)

            obj_id = p.loadURDF(obj_urdf)
            p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_orn)

        import ipdb
        ipdb.set_trace()

        p.disconnect()

    @staticmethod
    def save_result_stable(chair_stable_txt, stable_orn_list, stable_pos_list, stable_orn_eul_list):
        """Save the stable pose imagination result. The format is:
            qx, qy, qz, qw
            px, py, pz,
            ex, ey, ez
            ...
        """
        
        # Delete the file if it exist
        if os.path.exists(chair_stable_txt):
            os.remove(chair_stable_txt)

        with open(chair_stable_txt, 'w') as f:
            for i in range(len(stable_orn_list)):
                f.write(sci(stable_orn_list[i][0]) + ',')
                f.write(sci(stable_orn_list[i][1]) + ',')
                f.write(sci(stable_orn_list[i][2]) + ',')
                f.write(sci(stable_orn_list[i][3]) + '\n')
                f.write(sci(stable_pos_list[i][0]) + ',')
                f.write(sci(stable_pos_list[i][1]) + ',')
                f.write(sci(stable_pos_list[i][2]) + '\n')
                f.write(sci(stable_orn_eul_list[i][0]) + ',')
                f.write(sci(stable_orn_eul_list[i][1]) + ',')
                f.write(sci(stable_orn_eul_list[i][2]) + '\n')
        
