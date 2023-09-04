# Imagination base class for chair classification and upright pose imagination

from __future__ import print_function, division

import pybullet as p
import pybullet_data
import numpy as np

from utils import sci
# from utils import sci


class ImaginationMatrix(object):
    """Base class for sitting imagination with matrix pattern."""

    def __init__(self, agent_urdf, check_process=False, mp4_dir=None):
        """Constructor.

        Args:
            agent_urdf: path to the agent urdf file
            check_process: whether to visualize the process
            mp4_dir: directory to save the mp4 file of the imagination visualization
        """

        # Visualize the process in real time
        self.check_process = check_process

        # Hyperparameter
        self.simulation_iter = 500
        self.chair_rotate_iteration = None

        # Chair Matrix (x_num * y_num * episode_num = iteration)
        self.x_chair_num_functional = None
        self.y_chair_num_functional = None
        self.episode_num = None

        # assert self.x_chair_num_functional * self.y_chair_num_functional == self.chair_rotate_iteration_functional

        self.chair_adj_dist = 4  # distance between two adjacent chairs
        self.chair_id_list = []  # list to contain the id for each chair

        # Path to save the mp4
        self.mp4_dir = mp4_dir

        # Agent
        self.agent_urdf = agent_urdf
        self.agent_id_list = []
        self.human_ind_num = None

        # joint id
        self.root_id = 0
        self.chest_rotx_id = 1
        self.chest_roty_id = 2
        self.chest_rotz_id = 3
        self.neck_rotx_id = 4
        self.neck_rotz_id = 5
        self.left_shoulder_rotz_id = 6
        self.left_shoulder_roty_id = 7
        self.left_shoulder_rotx_id = 8
        self.right_shoulder_rotz_id = 9
        self.right_shoulder_roty_id = 10
        self.right_shoulder_rotx_id = 11
        self.right_hip_rotx_id = 12
        self.right_hip_rotz_id = 13
        self.right_knee_id = 14
        self.left_hip_rotx_id = 15
        self.left_hip_rotz_id = 16
        self.left_knee_id = 17

        ###### Joint ######
        self.normal_sitting = None
        self.normal_sitting_weight = None

        self.chest_rotz_limit = None
        self.chest_rotx_limit = None
        self.left_hip_rotx_limit = None
        self.right_hip_rotx_limit = None
        self.left_hip_rotz_limit = None
        self.right_hip_rotz_limit = None
        self.left_knee_limit = None
        self.right_knee_limit = None

        ###### Link ######
        self.normal_link_weight = None

        self.root_link_limit = None
        self.chest_link_limit = None
        self.left_hip_link_vertical_limit = None
        self.right_hip_link_vertical_limit = None

        ###### Chair ######
        self.chair_id_list = []
   

    def joint_angle_limit_check(self, sitting_joint):
        """Check the joint angle to increase the weight for punishment.
        
        Args:
            sitting_joint: numpy array of the agent's joint config.
        """
        
        raise NotImplementedError
        

    def absolute_link_limit_check(self, link_score):
        """Check the link rotation to increase the weight for punishment."""
        
        raise NotImplementedError

    def agent_drop_setup(self, agent_id, agent_start_pos, agent_start_orn):
        """Set up the agent for dropping.
        
        Args:
            agent_id: the id of the agent for setting up
            agent_start_pos: start position of the agent
            agent_start_orn: start orientation in quaternion
        """
        
        raise NotImplementedError

    def load_agent(self, agent_scale=1.0, pos=[0.0, 0.0, 0.0]):
        """Load an agent at the position. The mass of each link 
            is scaled to the correct ratio

        Args:
            agent_scale: scale of the agent.
            pos: position of the root link of the agent.
        """
        
        raise NotImplementedError

    def human_scale_func(self, chair_scale):
        """Human scale.
        
        Args:
            chair_scale: a number to scale the chair. This
                could be a length of the chair OBB.
        """
        
        raise NotImplementedError
        
    def visualize_result(self, chair_urdf, chair_sitting_orn,
                         chair_sitting_pos):
        """Visualize the result."""
        
        p.connect(p.GUI)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")

        p.resetDebugVisualizerCamera(3, 90, -45, chair_sitting_pos)

        chair_id = p.loadURDF(chair_urdf)
        p.resetBasePositionAndOrientation(chair_id, chair_sitting_pos,
                                          chair_sitting_orn)

        # import ipdb
        # ipdb.set_trace()

        p.disconnect()

    def get_com_pos_obb(self, obj_urdf):
        """Get the com position w.r.t. to the obb frame."""
        
        p.connect(p.DIRECT)
        obj_id = p.loadURDF(obj_urdf)
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        p.disconnect()

        return np.array(pos)

    @staticmethod
    def save_com(com_txt, com_pos):
        """Save the COM to file."""

        with open(com_txt, 'w') as f:
            f.write(
                sci(com_pos[0]) + " " + sci(com_pos[1]) + " " +
                sci(com_pos[2]) + "\n")
        print("Finish writing COM to {}".format(com_txt))

    def measure_agent(self):
        """Measure the height and weight of the agent."""
        agent_id = self.load_agent()

        head_aabb = p.getAABB(agent_id, self.neck_rotz_id)
        leg_aabb = p.getAABB(agent_id, self.left_knee_id)

        agent_height = head_aabb[1][1] - leg_aabb[0][1]
        print("Agent height: {}".format(agent_height))

        agent_mass = 0
        link_num = p.getNumJoints(agent_id)
        for link_idx in range(link_num):
            link_dynamic_info = p.getDynamicsInfo(agent_id, link_idx)
            agent_mass += link_dynamic_info[0]
        print("Agent mass: {}".format(agent_mass))

        p.removeBody(agent_id)

        return agent_height, agent_mass

    @staticmethod
    def save_com(com_txt, com_pos):
        """
        Save the COM to file
        """
        with open(com_txt, 'w') as f:
            f.write(
                sci(com_pos[0]) + " " + sci(com_pos[1]) + " " +
                sci(com_pos[2]) + "\n")
        print("Finish writing COM to {}".format(com_txt))