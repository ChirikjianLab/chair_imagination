#! /usr/bin/env python


import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from chair_functionality_matrix import ChairFunctionalityMatrix
from chair_stability_matrix import ChairStabilityMatrix
import preprocessing
import trimesh
import math
import csv

# from real_chair_imagination.utls import rotm2angle
from utils import rotm2angle, rotm2quat

# Evaluate pose for chair object
preprocess = True
stable_pose_imagination = True
functional_pose_imagination = True
benchmark = True
transform_folder = "transform_230831_100000"
# transform_folder = "transform"
rotation = "Exponential"
sitting_num_threshold = 0

agent_urdf = "./doc/humanoid_revolute_new.urdf"

save_csv_file = [
            "results/test_chairs_synthetic.csv",
                ]

obj_dir = [ 
            "/home/xin/Dropbox/chair_imagination_release/data/",
            ]


vhacd_dir = "/home/xin/lib/v-hacd/src/build/test" # directory where the vhacd file locates
meshlab_mlx = "./doc/TEMP3D_measure_geometry_scale.mlx"
meshlab_output_txt = "./doc/meshlab_output.txt"
meshlabserver_exe = "/home/xin/lib/meshlab-Meshlab-2020.03/distrib/meshlabserver"

# Chair stability
CS = ChairStabilityMatrix(check_process=False)

# Chair Functionality
CF = ChairFunctionalityMatrix(agent_urdf=agent_urdf, check_process=False)


for dir_idx, obj_folder in enumerate(obj_dir):
    obj_name_list = os.listdir(obj_folder)
    print("Total object number: ", len(obj_name_list))
    tested_obj_num = 0

    if 'non_chairs' in obj_folder:
        evaluate_pose = False
    else:
        evaluate_pose = True

    with open(save_csv_file[dir_idx], 'w') as csvfile:
        writer = csv.writer(csvfile)

        false_positive = []
        false_positive_num = 0

        for obj_name in obj_name_list:
            print(f"Object name: {obj_name}")

            csvwriterow = [obj_name]
            tested_obj_num += 1
            obj_mesh = os.path.join(obj_folder, obj_name, "origin", obj_name + ".obj")

            start_time = time.time()

            # Preprocessing
            # ------------------------------------------------------
            if preprocess:
                # OBB transform file
                transform_dir = os.path.join(obj_folder, str(obj_name), transform_folder)
                
                # Feb 09, 2020 added for evaluation
                if os.path.isdir(transform_dir):
                    print(obj_name + " has already been pre-processed!")
                else:
                    os.mkdir(transform_dir)

                # OBB transform
                transform_csv = os.path.join(transform_dir, obj_name + '_transform.csv') # CSV to save the transformation SE(3)
                obj_transform_mesh = os.path.join(transform_dir, obj_name + '_transform.obj') # Transformed mesh
                preprocessing.obb_transform(obj_mesh, obj_transform_mesh, transform_csv)

                # VHACD of transformed mesh
                obj_transform_vhacd = os.path.join(transform_dir, obj_name + '_transform_vhacd.obj') # VHACD mesh
                preprocessing.run_vhacd(vhacd_dir, obj_transform_mesh, obj_transform_vhacd)
                # import ipdb; ipdb.set_trace()
                # Meshlab compute properties and write urdf
                obj_transform_urdf = os.path.join(transform_dir, obj_name + '_transform.urdf')
                _ = preprocessing.meshlabcompute(obj_transform_vhacd, obj_transform_urdf, obj_name, 600, meshlabserver_exe, meshlab_mlx, meshlab_output_txt)

                if _ is None:
                    import ipdb; ipdb.set_trace()
                    continue

            preprocessing_time = time.time() - start_time

            # No preprocessing
            transform_dir = os.path.join(obj_folder, str(obj_name), transform_folder)
            obj_transform_urdf = os.path.join(transform_dir, obj_name + '_transform.urdf')
            obj_transform_mesh = os.path.join(transform_dir, obj_name + '_transform.obj') # Transformed mesh
            # # ------------------------------------------------------

            # Stable pose imagination
            # ------------------------------------------------------
            if stable_pose_imagination:
                stable_orn, stable_pos, stable_orn_eul = CS.get_stable_pose(obj_transform_urdf, obj_transform_mesh)
                # stable_orn, stable_pos, stable_orn_eul = CS.get_stable_pose_baseline(obj_transform_urdf, obj_transform_mesh)
                stable_pose_imagination_time = time.time() - start_time - preprocessing_time
            # ------------------------------------------------------

            # Functional pose imagination
            # ------------------------------------------------------
            if functional_pose_imagination:
                sitting_correct, total_sitting_score = CF.get_functional_pose(obj_transform_urdf, obj_transform_mesh, stable_orn_eul, stable_pos)
                functional_pose_imagination_time = time.time() - start_time - preprocessing_time - stable_pose_imagination_time
            # ------------------------------------------------------

            if benchmark:
                total_imagination_time = time.time() - start_time
                print("Preprocessing time: {}".format(preprocessing_time))
                print("Stable pose imagination time: {}".format(stable_pose_imagination_time))
                print("Functional pose imagination time: {}".format(functional_pose_imagination_time))
                print("Total imagination time: {}".format(total_imagination_time))
                csvwriterow.append(preprocessing_time)
                csvwriterow.append(stable_pose_imagination_time)
                csvwriterow.append(functional_pose_imagination_time)
                csvwriterow.append(total_imagination_time)

                if sitting_correct and sitting_correct[1] >= sitting_num_threshold:
                    print("This obect is chairish enough!")
                    # correct sitting number
                    csvwriterow.append(sitting_correct[1])
                    csvwriterow.append(total_sitting_score)
                    csvwriterow.append("classification")
                    csvwriterow.append(1)

                    if evaluate_pose:
                        ##############################################################################
                        # benchmark
                        obj_transform_matrix_csv = os.path.join(obj_folder, obj_name, transform_folder, obj_name + "_transform.csv")
                        obj_origin_urdf = os.path.join(obj_folder, obj_name, "origin", obj_name + ".urdf")
                        obj_origin_mesh = os.path.join(obj_folder, obj_name, "origin", obj_name + ".obj")
                        obb_transform_matrix = []
                        with open(obj_transform_matrix_csv) as csvfile:
                            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                            for idx, row in enumerate(reader):
                                # print("Row: ", row)
                                if idx < 3:
                                    obb_transform_matrix.append(row[:3])
                        obb_transform_matrix = np.array(obb_transform_matrix)
                
                        ##############################################################################

                        correct_pose_wrt_obb = p.getMatrixFromQuaternion(stable_orn[sitting_correct[0]])
                        correct_pose_wrt_obb = np.reshape(np.array(correct_pose_wrt_obb), (3, 3))
                        correct_pose_wrt_origin_matrix = np.dot(correct_pose_wrt_obb, obb_transform_matrix)
                        print("Correct orn w.r.t origin: ", correct_pose_wrt_origin_matrix)
                        correct_pose_wrt_origin_qua = rotm2quat(correct_pose_wrt_origin_matrix)
            
                        physicsClient_evl = p.connect(p.DIRECT)
                        p.setAdditionalSearchPath(pybullet_data.getDataPath())
                        p.setGravity(0, 0, -10)

                        plane_id_evl = p.loadURDF("plane.urdf")
                        p.changeDynamics(plane_id_evl, -1, restitution=0.9)

                        chair_id_evl = p.loadURDF(obj_origin_urdf)
                        p.resetBasePositionAndOrientation(chair_id_evl, posObj=[0, 0, 0], ornObj=correct_pose_wrt_origin_qua)

                        # Drop Height for finding the translation along the z-axis               
                        obj_mesh = trimesh.load(obj_transform_mesh)
                        obj_bbox = obj_mesh.extents
                        obj_extent = math.sqrt(obj_bbox[0] * obj_bbox[0] +
                                                obj_bbox[1] * obj_bbox[1] +
                                                obj_bbox[2] * obj_bbox[2])
                        chair_drop_height = obj_extent / 2 + 0.05

                        p.resetBasePositionAndOrientation(chair_id_evl, posObj=[0, 0, chair_drop_height],
                                                        ornObj=correct_pose_wrt_origin_qua)

                        for j in range(500):
                            p.stepSimulation()
                            if True:
                                time.sleep(1. / 240.)
                        chair_evl_pos, chair_evl_orn = p.getBasePositionAndOrientation(chair_id_evl)
                        p.disconnect()
                        chair_evl_rotm = np.array(p.getMatrixFromQuaternion(chair_evl_orn)).reshape(3, 3)

                        
                        ##############################################################################
                        # Benchmark
                        obj_annotation_csv = os.path.join(obj_folder, obj_name, "origin", obj_name + "_annotation.csv")
                        annotation_pos_z = 0.0
                        annotation_rotm = []
                        with open(obj_annotation_csv, 'r') as anncsv:
                            reader = csv.reader(anncsv, quoting=csv.QUOTE_NONNUMERIC)
                            for idx, row in enumerate(reader):
                                if idx == 0:
                                    annotation_pos_z = row[-1]
                                else:
                                    annotation_rotm.append(row)
                        annotation_rotm = np.array(annotation_rotm)
                        ##############################################################################

                        z_diff = abs(annotation_pos_z - chair_evl_pos[2])
                        z_theta, _, _, axis_z = rotm2angle(np.matmul(annotation_rotm, chair_evl_rotm.transpose()))
                    
                        print(f"z_diff: {z_diff}")
                        print(f"z_theta: {z_theta}")
                        print(f"axis_z: {axis_z}")

                        csvwriterow.append('pose')

                        if z_diff < 0.01 and (abs(axis_z) > 0.99 or abs(z_theta) < 0.01):
                            print("The upright pose is correct!")
                            csvwriterow.append(1)
                        else:
                            print("The upright pose is incorrect!!!!!")
                            csvwriterow.append(0)
                        ###################################################
                else:
                    print("This object is not chairish enough!")
                    # correct sitting number
                    if sitting_correct:
                        csvwriterow.append(sitting_correct[1])
                    else:
                        csvwriterow.append(0)
                    # sitting score
                    csvwriterow.append(total_sitting_score)
                    # chair classification: 0-nonchair, 1-chair
                    csvwriterow.append("classification")
                    csvwriterow.append(0)

                    if evaluate_pose:
                        csvwriterow.append("pose")
                        csvwriterow.append("0")

                    false_positive.append(obj_name)
                    false_positive_num += 1
                    print("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}")
                print(obj_name)
                print("Finish!")
                print("Tested object number: ", tested_obj_num)
                print("False positive: ", false_positive)
                print("False positive number: ", false_positive_num)

                writer.writerows([csvwriterow])
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")