"""
Functionality imagination of an object as a chair.
Given a set of stable poses of the object, the code simulate different poses
and find the one which affords sitting. If such a pose exists, the object is
classified as a chair.
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
import os
import trimesh
from imagination import ImaginationMatrix


class ChairFunctionalityMatrix(ImaginationMatrix):
    """Class to imagine the functional pose of the chair for classification."""

    def __init__(self, agent_urdf, check_process=False, mp4_dir=None):
        """Constructor.

        Args:
            agent_urdf: path to the agent urdf file
            check_process: whether to visualize the process
            mp4_dir: directory to save the mp4 file of the imagination visualization
        """

        ImaginationMatrix.__init__(self, agent_urdf, check_process, mp4_dir)
        
        self.check_process = check_process
        self.agent_urdf = agent_urdf

        # Hyperparameter
        self.simulation_iter = 500
        self.chair_rotate_iteration = 18
        self.x_chair_num_functional = 3
        self.y_chair_num_functional = 3
        self.episode_num = 2

        self.normal_sitting = np.array([0.0, 0.0, 0.0, math.pi / 6,
                                        0.0, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 5 * math.pi / 12, -math.pi / 2,
                                        0.0, 5 * math.pi / 12, -math.pi / 2])

        self.normal_sitting_weight = np.array([0.0, 0.5, 0.5, 1.0,
                                               0.0, 0.0,
                                               0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0,
                                               0.5, 0.8, 0.5,
                                               0.5, 0.8, 0.5])

        self.normal_link_weight = np.array([1.0, 1.0, 0.5, 0.5])

        self.chest_rotz_limit = np.array([-math.pi / 6, math.pi / 6])
        self.chest_rotx_limit = np.array([-math.pi / 4, math.pi / 4])
        self.left_hip_rotx_limit = np.array([-math.pi / 6, math.pi / 6])
        self.right_hip_rotx_limit = np.array([-math.pi / 6, math.pi / 6])
        self.left_hip_rotz_limit = np.array([-math.pi / 4, math.pi / 3])
        self.right_hip_rotz_limit = np.array([-math.pi / 4, math.pi / 3])
        self.left_knee_limit = np.array([-math.pi / 6, math.pi / 3])
        self.right_knee_limit = np.array([-math.pi / 6, math.pi / 3])

        self.root_link_limit = 1 - math.cos(math.pi / 4)
        self.chest_link_limit = 1 - math.cos(math.pi / 4)
        self.left_hip_link_vertical_limit = 1 - math.cos(math.pi / 4)
        self.right_hip_link_vertical_limit = 1 - math.cos(math.pi / 4)

        self.joint_angle_score_max = 2.0
        self.absolute_link_state_score_max = 0.5
        self.total_contact_num_min = 5

        self.hip_height_max = 1.3
        self.hip_height_min = 0.1

        self.human_ind_num = 3
        self.human_ind_ratio = 0.1

        # Friction
        self.chair_friction_coeff = 1.0
    
    def joint_angle_limit_check(self, sitting_joint):
        """Check the joint angle to increase the weight for punishment.
        
        Args:
            sitting_joint: numpy array of the agent's joint config.
        """

        curr_sitting_weight = np.copy(self.normal_sitting_weight)

        if (sitting_joint[self.chest_rotz_id] - self.normal_sitting[self.chest_rotz_id]) < self.chest_rotz_limit[0] or \
                (sitting_joint[self.chest_rotz_id] - self.normal_sitting[self.chest_rotz_id]) > self.chest_rotz_limit[-1]:
            curr_sitting_weight[self.chest_rotz_id] *= 3

        if (sitting_joint[self.chest_rotx_id] - self.normal_sitting[self.chest_rotx_id]) < self.chest_rotx_limit[0] or \
                (sitting_joint[self.chest_rotx_id] - self.normal_sitting[self.chest_rotx_id]) > self.chest_rotx_limit[-1]:
            curr_sitting_weight[self.chest_rotx_id] *= 3

        if (sitting_joint[self.left_hip_rotx_id] - self.normal_sitting[self.left_hip_rotx_id]) < self.left_hip_rotx_limit[0] or \
                (sitting_joint[self.left_hip_rotx_id] - self.normal_sitting[self.left_hip_rotx_id]) > self.left_hip_rotx_limit[-1]:
            curr_sitting_weight[self.left_hip_rotx_id] *= 3

        if (sitting_joint[self.left_hip_rotz_id] - self.normal_sitting[self.left_hip_rotz_id]) < self.left_hip_rotz_limit[0] or \
                (sitting_joint[self.left_hip_rotz_id] - self.normal_sitting[self.left_hip_rotz_id]) > self.left_hip_rotz_limit[-1]:
            curr_sitting_weight[self.left_hip_rotz_id] *= 3

        if (sitting_joint[self.right_hip_rotx_id] - self.normal_sitting[self.right_hip_rotx_id]) < self.right_hip_rotx_limit[0] or \
                (sitting_joint[self.right_hip_rotx_id] - self.normal_sitting[self.right_hip_rotx_id]) > self.right_hip_rotx_limit[-1]:
            curr_sitting_weight[self.right_hip_rotx_id] *= 3

        if (sitting_joint[self.right_hip_rotz_id] - self.normal_sitting[self.right_hip_rotz_id]) < self.right_hip_rotz_limit[0] or \
                (sitting_joint[self.right_hip_rotz_id] - self.normal_sitting[self.right_hip_rotz_id]) > self.right_hip_rotz_limit[-1]:
            curr_sitting_weight[self.right_hip_rotz_id] *= 3

        if (sitting_joint[self.left_knee_id] - self.normal_sitting[self.left_knee_id]) < self.left_knee_limit[0] or \
                (sitting_joint[self.left_knee_id] - self.normal_sitting[self.left_knee_id]) > self.left_knee_limit[-1]:
            curr_sitting_weight[self.left_knee_id] *= 3

        if (sitting_joint[self.right_knee_id] - self.normal_sitting[self.right_knee_id]) < self.right_knee_limit[0] or \
                (sitting_joint[self.right_knee_id] - self.normal_sitting[self.right_knee_id]) > self.right_knee_limit[-1]:
            curr_sitting_weight[self.right_knee_id] *= 3

        return curr_sitting_weight

    def absolute_link_limit_check(self, link_score):
        """Check the link rotation to increase the weight for punishment."""

        curr_link_weight = np.copy(self.normal_link_weight)

        if link_score[0] > self.root_link_limit:
            curr_link_weight[0] *= 3
        if link_score[1] > self.chest_link_limit:
            curr_link_weight[1] *= 3
        if link_score[2] > self.right_hip_link_vertical_limit:
            curr_link_weight[2] *= 3
        if link_score[3] > self.left_hip_link_vertical_limit:
            curr_link_weight[3] *= 3

        return curr_link_weight

    def agent_drop_setup(self, humanoid_id, humanoid_start_pos, humanoid_start_orn):
        """Set up the agent for dropping.
        
        Args:
            agent_id: the id of the agent for setting up
            agent_start_pos: start position of the agent
            agent_start_orn: start orientation in quaternion
        """

        p.resetBasePositionAndOrientation(humanoid_id, humanoid_start_pos, humanoid_start_orn)

        # Chest
        p.resetJointState(humanoid_id, self.chest_rotx_id, 0.0)
        p.resetJointState(humanoid_id, self.chest_roty_id, 0.0)
        p.resetJointState(humanoid_id, self.chest_rotz_id, math.pi / 10)
        p.setJointMotorControl2(humanoid_id, self.chest_rotx_id, p.POSITION_CONTROL, targetPosition=0.0, force=1.0)

        # Neck
        p.resetJointState(humanoid_id, self.neck_rotz_id, 0.0)
        p.resetJointState(humanoid_id, self.neck_rotx_id, 0.0)
        p.setJointMotorControl2(humanoid_id, self.neck_rotz_id, p.POSITION_CONTROL, targetPosition=0.0,
                                force=0.0)
        p.setJointMotorControl2(humanoid_id, self.neck_rotx_id, p.POSITION_CONTROL, targetPosition=0.0,
                                force=0.0)
        # Shoulder
        p.resetJointState(humanoid_id, self.right_shoulder_rotz_id, 0.0)
        p.resetJointState(humanoid_id, self.left_shoulder_rotz_id, 0.0)
        p.resetJointState(humanoid_id, self.right_shoulder_roty_id, 0.0)
        p.resetJointState(humanoid_id, self.left_shoulder_roty_id, 0.0)
        p.resetJointState(humanoid_id, self.right_shoulder_rotx_id, 0.0)
        p.resetJointState(humanoid_id, self.left_shoulder_rotx_id, 0.0)
        p.setJointMotorControl2(humanoid_id, self.right_shoulder_rotz_id, p.POSITION_CONTROL, 0.0, force=0.0)
        p.setJointMotorControl2(humanoid_id, self.left_shoulder_rotz_id, p.POSITION_CONTROL, 0.0, force=0.0)

        # Hip
        p.resetJointState(humanoid_id, self.right_hip_rotx_id, 0.0)
        p.resetJointState(humanoid_id, self.right_hip_rotz_id, math.pi / 2)
        p.resetJointState(humanoid_id, self.left_hip_rotx_id, 0.0)
        p.resetJointState(humanoid_id, self.left_hip_rotz_id, math.pi / 2)
        p.setJointMotorControl2(humanoid_id, self.left_hip_rotz_id, p.POSITION_CONTROL,
                                targetPosition=math.pi / 2, force=0.0)
        p.setJointMotorControl2(humanoid_id, self.left_hip_rotx_id, p.POSITION_CONTROL, targetPosition=0.0,
                                force=0.0)
        p.setJointMotorControl2(humanoid_id, self.right_hip_rotz_id, p.POSITION_CONTROL,
                                targetPosition=math.pi / 2, force=0.0)
        p.setJointMotorControl2(humanoid_id, self.right_hip_rotx_id, p.POSITION_CONTROL, targetPosition=0.0,
                                force=0.0)
        # Knee
        p.resetJointState(humanoid_id, self.left_knee_id, -math.pi / 4)
        p.resetJointState(humanoid_id, self.right_knee_id, -math.pi / 4)
        p.setJointMotorControl2(humanoid_id, self.left_knee_id, p.POSITION_CONTROL,
                                targetPosition=-math.pi / 2, force=0.0)
        p.setJointMotorControl2(humanoid_id, self.right_knee_id, p.POSITION_CONTROL,
                                targetPosition=-math.pi / 2, force=0.0)

    def get_link_scores(self, agent_id):
        """Get the link scores of each link."""

        z_axis = np.array([[0], [0], [1]])
        # Root link
        root_link_state = p.getLinkState(agent_id, self.root_id)
        root_link_quat = root_link_state[1]
        root_link_matrix = np.linalg.inv(
        np.reshape(np.array(p.getMatrixFromQuaternion(root_link_quat)), (3, 3)))
        root_link_vec = root_link_matrix[1]
        root_link_state_score = abs(np.dot(root_link_vec, z_axis) - 1)
        # print("Root link matrix: ", root_link_matrix)
        # print("Root link score: ", root_link_state_score)

        # Chest link
        chest_link_state = p.getLinkState(agent_id, self.chest_rotz_id)
        chest_link_quat = chest_link_state[1]
        chest_link_matrix = np.linalg.inv(
            np.reshape(np.array(p.getMatrixFromQuaternion(chest_link_quat)), (3, 3)))
        chest_link_vec = chest_link_matrix[1]
        chest_link_state_score = abs(np.dot(chest_link_vec, np.array([-0.5, 0, 0.866])) - 1)
        # print("Chest link matrix: ", chest_link_matrix)
        # print("Chest link score: ", chest_link_state_score)

        # Right hip link
        right_hip_link_state = p.getLinkState(agent_id, self.right_hip_rotz_id)
        right_hip_link_quat = right_hip_link_state[1]
        right_hip_link_matrix = np.linalg.inv(
            np.reshape(np.array(p.getMatrixFromQuaternion(right_hip_link_quat)), (3, 3)))
        right_hip_link_vec_vertical = right_hip_link_matrix[0]
        right_hip_link_state_score = abs(np.dot(right_hip_link_vec_vertical, z_axis) - 1)
        # print("Right hip link matrix rotz")
        # print(right_hip_link_matrix)
        # print("Right hip link score: ", right_hip_link_state_score)

        # Left hip link
        left_hip_link_state = p.getLinkState(agent_id, self.left_hip_rotz_id)
        left_hip_link_qua = left_hip_link_state[1]
        left_hip_link_matrix = np.linalg.inv(
            np.reshape(np.array(p.getMatrixFromQuaternion(left_hip_link_qua)), (3, 3)))
        left_hip_link_vec_vertical = left_hip_link_matrix[0]
        left_hip_link_state_score = abs(np.dot(left_hip_link_vec_vertical, z_axis) - 1)
        # print("Left hip link matrix rotz")
        # print(left_hip_link_matrix)
        # print("Left hip link score: ", left_hip_link_state_score)

        link_score = np.array([root_link_state_score, chest_link_state_score, 
            right_hip_link_state_score, left_hip_link_state_score])
        
        return link_score

    def human_scale_func(self, chair_scale):
        """Human scale.
        
        Args:
            chair_scale: a number to scale the chair. This
                could be a length of the chair OBB.
        """

        human_scale = 0.22 + (chair_scale - 0.60) * 0.04 / 0.1

        if human_scale > 0.36:
            human_scale = 0.36  # about 2.5m

        if human_scale < 1/8:
            human_scale = 1/8  # about 1.3m

        return human_scale
    
    def load_agent(self, agent_scale=1.0, pos=[0.0, 0.0, 0.0]):
        """
        Load an agent at the position.
        The mass of each link is scaled to the correct ratio

        @type  pos: list of 3 float
        @param pos: position of the root link of the agent
        """

        agent_id = p.loadURDF(self.agent_urdf,
                              basePosition=pos,
                              globalScaling=agent_scale)
        p.changeDynamics(agent_id, -1, restitution=0.0)
        p.changeDynamics(agent_id, self.chest_rotz_id, lateralFriction=0.3)
        p.changeDynamics(agent_id, self.neck_rotz_id, lateralFriction=0.3)

        for link_idx in range(p.getNumJoints(agent_id)):
            p.changeDynamics(agent_id, link_idx, restitution=0.0)
            p.changeDynamics(agent_id, link_idx, lateralFriction=1.0)
        return agent_id

    def get_functional_pose(self, obj_urdf, obj_transform_mesh, stable_orn, stable_pos):
        """Find the functional pose among a list of equivalently stable poses.

        Args:
            obj_urdf: urdf of the OBB transfrormed object for imagination
            obj_transform_mesh: OBB transformed object mesh
            stable_orn: a list of stable orientation of the object
            stable_pos: a list of stable position of the object. It corresponds 
                with stable_orn.
        Returns:
            functional_pose_orn: functional pose orientation in quaternion (x, y, z, w)
            functional_pose_pos: functional pose position
            sitting_pose_orn: NotImplemented
            sitting_pose_pos: NotImplemented
        """

        # Set up the environment
        if self.check_process:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(3, 90, -45, [3, 2, 0])

        # Load the plane
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, restitution=0.0)

        # The human scale is determined by the scale of the chair
        trimesh_mesh = trimesh.load(obj_transform_mesh)
        chair_extents = trimesh_mesh.extents
        chair_extents_argsort = np.argsort(np.array(chair_extents))
        chair_scale = chair_extents[chair_extents_argsort[1]]
        
        # The human indentation in the sitting imagination
        human_ind = self.human_ind_ratio * chair_extents[chair_extents_argsort[0]]

        # Load the humanoid model
        human_scale = self.human_scale_func(chair_scale)
        print(f"human_scale: {human_scale}")
        
        # Load chairs
        assert self.x_chair_num_functional > 0
        assert self.y_chair_num_functional > 0
        for x_idx in range(self.x_chair_num_functional):
            for y_idx in range(self.y_chair_num_functional):
                chair_start_x = x_idx * self.chair_adj_dist
                chair_start_y = y_idx * self.chair_adj_dist

                # Set the OBB center to positions
                chair_id = p.loadURDF(
                    obj_urdf,
                    basePosition=[chair_start_x, chair_start_y, 0.0],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]))

                p.changeDynamics(chair_id, -1, restitution=0.0)
                p.changeDynamics(chair_id, -1, lateralFriction=self.chair_friction_coeff)

                self.chair_id_list.append(chair_id)

        # Load agent
        for x_idx in range(self.x_chair_num_functional):
            for y_idx in range(self.y_chair_num_functional):
                chair_start_x = x_idx * self.chair_adj_dist
                chair_start_y = y_idx * self.chair_adj_dist
                
                agent_id = self.load_agent(human_scale, [chair_start_x, chair_start_y, 10.0])

                self.agent_id_list.append(agent_id)

        # TODO: Remove this part
        chair_id = p.loadURDF(obj_urdf)
        chair_curr_pos, _ = p.getBasePositionAndOrientation(chair_id)
        chair_curr_pos = np.transpose(np.array(chair_curr_pos))
        p.removeBody(chair_id)

        sitting_correct = []
        max_sitting_config_score = 0

        # Save mp4 video
        if (self.mp4_dir is not None) and (self.check_process):
            obj_name = obj_urdf.split('/')[-1].split('.')[0]
            mp4_file_name = obj_name + "_chair_imagination.mp4"
            mp4_file_path = os.path.join(self.mp4_dir, mp4_file_name)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, mp4_file_path)

        for chair_stable_idx, chair_stable_orn in enumerate(stable_orn):
            chair_stable_pos = stable_pos[chair_stable_idx]
            chair_start_pos = [0.0, 0.0, chair_stable_pos[-1]]

            sitting_num = 0
            sitting_height = 0
            sitting_joint_score = 0
            sitting_absolute_link_score = 0
            sitting_config_score = 0

            p.setGravity(0, 0, -10)

            for ep_idx in range(self.episode_num):
                for j in range(self.human_ind_num):
                    for x_idx in range(self.x_chair_num_functional):
                        for y_idx in range(self.y_chair_num_functional):
                            chair_rotate_idx = y_idx + x_idx * self.y_chair_num_functional

                            chair_z_axis_angle = -(y_idx + x_idx * self.y_chair_num_functional + \
                                ep_idx * self.x_chair_num_functional * self.y_chair_num_functional) * 2 * math.pi / self.chair_rotate_iteration
                            chair_start_orn = [chair_stable_orn[0], chair_stable_orn[1], chair_z_axis_angle]

                            chair_stable_orn_mat = np.reshape(np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(chair_start_orn))), (3, 3))
                            chair_initial_center = np.copy(chair_curr_pos)
                            chair_center = np.dot(chair_stable_orn_mat, chair_initial_center)
                            chair_start_pos[0] = chair_center[0] + x_idx * self.chair_adj_dist
                            chair_start_pos[1] = chair_center[1] + y_idx * self.chair_adj_dist

                            p.resetBasePositionAndOrientation(self.chair_id_list[chair_rotate_idx], 
                                                            chair_start_pos,
                                                            p.getQuaternionFromEuler(chair_start_orn))

                            chair_aabb = p.getAABB(self.chair_id_list[chair_rotate_idx])
                            chair_bbox_largest = chair_aabb[1][2]

                            # humanoid facing x direction
                            humanoid_id = self.agent_id_list[chair_rotate_idx]
                            humanoid_start_pos = ((j) * human_ind + x_idx * self.chair_adj_dist, 
                                                y_idx * self.chair_adj_dist, 
                                                chair_bbox_largest + 0.15)
                            humanoid_start_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
                            self.agent_drop_setup(humanoid_id, humanoid_start_pos, humanoid_start_orn)
                           
                    for i in range(int(self.simulation_iter)):
                        p.stepSimulation()

                    # Check each sitting
                    for x_idx in range(self.x_chair_num_functional):
                        for y_idx in range(self.y_chair_num_functional):
                            chair_rotate_idx = y_idx + x_idx * self.y_chair_num_functional
                            agent_id = self.agent_id_list[chair_rotate_idx]
                            chair_id = self.chair_id_list[chair_rotate_idx]

                            #######################################
                            # Joint Angle Score
                            sitting = []
                            for joint_idx in range(p.getNumJoints(agent_id)):
                                joint_state = p.getJointState(agent_id, joint_idx)
                                sitting.append(joint_state[0])
                            sitting = np.array(sitting)

                            sitting_weight = self.joint_angle_limit_check(sitting)
                            distance_wrt_normal_sitting = np.multiply(sitting_weight,
                                                                    np.subtract(sitting, self.normal_sitting))
                            joint_angle_score = np.sum(np.absolute(distance_wrt_normal_sitting))

                            #######################################
                            # Link Score
                            if joint_angle_score < self.joint_angle_score_max:
                                link_score = self.get_link_scores(agent_id)
                                curr_link_weight = self.absolute_link_limit_check(link_score)
                                absolute_link_state_score = np.sum(np.multiply(link_score, curr_link_weight))[0]

                            #######################################
                            # Contact points
                                if absolute_link_state_score < self.absolute_link_state_score_max:
                                    head_contact_point = p.getContactPoints(chair_id, agent_id, -1, self.neck_rotz_id)
                                    head_contact_num = len(head_contact_point)
                                    # print("head contact point number: ", head_contact_num)
                                    chest_contact_point = p.getContactPoints(chair_id, agent_id, -1, self.chest_rotz_id)
                                    chest_contact_num = len(chest_contact_point)
                                    # print("chest contact point number: ", chest_contact_num)
                                    left_hip_contact_point = p.getContactPoints(chair_id, agent_id, -1, 
                                        self.left_hip_rotz_id)
                                    left_hip_contact_num = len(left_hip_contact_point)
                                    # print("left hip contact point number: ", left_hip_contact_num)
                                    right_hip_contact_point = p.getContactPoints(chair_id, agent_id, -1,
                                        self.right_hip_rotz_id)
                                    right_hip_contact_num = len(right_hip_contact_point)
                                    # print("right hip contact point number: ", right_hip_contact_num)

                                    left_shoulder_contact_point = p.getContactPoints(chair_id, agent_id, -1, 
                                        self.left_shoulder_rotx_id)
                                    left_shoulder_contact_num = len(left_shoulder_contact_point)
                                    right_shoulder_contact_point = p.getContactPoints(chair_id, agent_id, -1,
                                        self.right_shoulder_rotx_id)
                                    right_shoulder_contact_num = len(right_shoulder_contact_point)

                                    # total_contact_number = chest_contact_num + head_contact_num + \
                                    #     left_hip_contact_num + right_hip_contact_num
                                    total_contact_number = chest_contact_num + head_contact_num + \
                                        left_hip_contact_num + right_hip_contact_num + \
                                        left_shoulder_contact_num + right_shoulder_contact_num

                            #######################################
                            # Sitting height
                                    # if (chest_contact_num + head_contact_num) * left_hip_contact_num * right_hip_contact_num > 0 \
                                    #     and total_contact_number >= self.total_contact_num_min:
                                    upper_body_contact_num = chest_contact_num + head_contact_num + left_hip_contact_num + right_shoulder_contact_num
                                    if upper_body_contact_num * left_hip_contact_num * right_hip_contact_num > 0 \
                                        and total_contact_number >= self.total_contact_num_min:
                                        left_hip_height_list = np.array([left_hip_contact_point[i][6][-1]
                                            for i, _ in enumerate(left_hip_contact_point)])
                                        right_hip_height_list = np.array([right_hip_contact_point[i][6][-1]
                                            for i, _ in enumerate(right_hip_contact_point)])

                                        hip_height = 0.5 * (np.average(left_hip_height_list) + np.average(right_hip_height_list))
                                        
                                        if hip_height > self.hip_height_min and hip_height < self.hip_height_max:
                                            # It is classify as a sitting
                                            sitting_num += 1
                                            sitting_height = (sitting_height * (sitting_num - 1) + hip_height) / sitting_num
                                            sitting_joint_score = (sitting_joint_score * (
                                                        sitting_num - 1) + joint_angle_score) / sitting_num
                                            sitting_absolute_link_score = (sitting_absolute_link_score * (
                                                        sitting_num - 1) + absolute_link_state_score) / sitting_num
                                            sitting_config_score += 1 / (joint_angle_score * absolute_link_state_score)
            if sitting_num < 1:
                continue

            avg_config_score = sitting_config_score / sitting_num
            print(f'Avg Config Score: {avg_config_score}')
            print(f'Avg Sitting Height: {sitting_height}')
            print(f'Stting Num: {sitting_num}')

            if not sitting_correct:
                sitting_correct.append(chair_stable_idx)
                sitting_correct.append(sitting_num)
                sitting_correct.append(sitting_height)
                sitting_correct.append(sitting_config_score)
                max_sitting_config_score = sitting_config_score
            elif (sitting_num / sitting_correct[1]) * (sitting_height / sitting_correct[2]) > 1:
                sitting_correct[0] = chair_stable_idx
                sitting_correct[1] = sitting_num
                sitting_correct[2] = sitting_height
                sitting_correct[3] = sitting_config_score
                max_sitting_config_score = sitting_config_score

        p.disconnect()
        print("Sitting correct: ", sitting_correct)
        print("Sitting config score: ", max_sitting_config_score)
        
        return sitting_correct, max_sitting_config_score