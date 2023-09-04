"""
Preprocessing the raw obj file for imagination.

OBB transform the arbitrarily oriented object
VHACD for the OBB transformed object
Generate the URDF file for the OBB object

"""

import os
import time
import subprocess
import trimesh
import csv
import meshlabxml as mlx
import numpy as np
import re

root_dir = "/home/xin/panda_ws/src/chair_imagination"

vhacd_dir = "/home/xin/lib/v-hacd/src/build/test"
meshlabserver_exe = "/home/xin/lib/meshlab-Meshlab-2020.03/distrib/meshlabserver"
meshlab_mlx = os.path.join(root_dir, "doc/TEMP3D_measure_geometry_scale.mlx")
meshlab_output_txt = os.path.join(root_dir, "doc/meshlab_output.txt")

meshlabserver_exe = "/home/xin/lib/meshlab-Meshlab-2020.03/distrib/meshlabserver"

def obb_transform(obj_path, obj_transform_path, csv_path):
    """
    Apply OBB transformation on the object
    Save the OBB SE(3) transform in a csv file

    Args:
    -- obj_path: path to the object obj file
    -- obj_transform_path: path for saving the transformed obj file
    -- csv_path: path to the csv for saving the SE(3) of obb transofrm
    """
    origin_obj = trimesh.load(obj_path)
    obb_transform = origin_obj.apply_obb()

    # Write the SE(3) of the OBB transform
    with open(csv_path, 'w+', newline = '\n') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows([obb_transform[0]])
        writer.writerows([obb_transform[1]])
        writer.writerows([obb_transform[2]])
        writer.writerows([obb_transform[3]])
    writeFile.close()

    # Save the transformed model
    origin_obj.export(obj_transform_path, "obj")


def run_vhacd(vhacd_dir, input_file, output_file, log='log.txt', resolution=1000000, depth=20, concavity=0.0025,planeDownsampling=8, 
    convexhullDownsampling=8, alpha=0.05, beta=0.05, gamma=0.00125,pca=0, mode=0, maxNumVerticesPerCH=32, 
    minVolumePerCH=0.0001, convexhullApproximation=1, oclDeviceID=2):
    """
    The wrapper function to run the vhacd convex decomposition.

    #// --input camel.off --output camel_acd.wrl --log log.txt --resolution 1000000 --depth 20 --concavity 0.0025 --planeDownsampling 4 --convexhullDownsampling 4 --alpha 0.05 --beta 0.05 --gamma 0.00125 
    # --pca 0 --mode 0 --maxNumVerticesPerCH 256 --minVolumePerCH 0.0001 --convexhullApproximation 1 --oclDeviceID 2
    """

    vhacd_executable = os.path.join(vhacd_dir, 'testVHACD')
    if not os.path.isfile(vhacd_executable):
        print (vhacd_executable)
        raise ValueError('vhacd executable not found, have you compiled it?')

    cmd = "cd %s && %s --input %s --output %s --log %s --resolution %s --depth %s --concavity %s --planeDownsampling %s --convexhullDownsampling %s --alpha %s --beta %s --gamma %s \
        --pca %s --mode %s --maxNumVerticesPerCH %s --minVolumePerCH %s --convexhullApproximation %s --oclDeviceID %s" %(vhacd_dir, vhacd_executable, input_file, output_file, log, resolution,
        depth, concavity, planeDownsampling, convexhullDownsampling, alpha, beta, gamma, pca, mode, maxNumVerticesPerCH, minVolumePerCH, convexhullApproximation, oclDeviceID)

    print ("cmd:\n", cmd)

    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True)
    print ("started subprocess, waiting for V-HACD to finish")
    process.wait()
    elapsed = time.time() - start_time

    print ("V-HACD took %d seconds" %(elapsed))


def parse_float(string):
    """
    Parse the float within a string.
    
    Args:
    -- string: string input
    Retunrs:
    -- float_list: list of all floats in order
    """
    return np.array([float(i) for i in re.findall(r"[-+]?\d*\.\d+|\d+", string)])


def meshlabcompute(obj_path, urdf_path, obj_name, obj_density,
                   meshlabserver_exe, meshlab_mlx, meshlab_output_txt):
    """
    Compute the mass, mass center, and inertia for the object
    Install meshlab 2020.03 from source
    Args:
        obj_path: path to the obj file
        urdf_path: path to save the urdf file
        obj_density: the density of the object in kg/m^3.
        meshlabserver_exe: path to the meshlabserver executable
        meshlab_mlx: meshlab mlx file to compute the physics properties of the object
        meshlab_output_txt: output txt to save the meshlab result
    """
    if os.path.exists(meshlab_output_txt):
        os.remove(meshlab_output_txt)

    cmd = meshlabserver_exe + ' '
    cmd += '-i' + ' \"' + obj_path + '\" '
    cmd += '-s' + ' \"' + meshlab_mlx + '\" '
    cmd += '-l' + ' \"' + meshlab_output_txt + '\"'

    # print(cmd)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    # Parse the log file to get the volume, center of mass, and inertia
    with open(meshlab_output_txt, 'r') as file1:
        log = file1.readlines()

        properties = {}

        for idx, line in enumerate(reversed(log)):
            if ("Mesh Volume" in line) and (not "mass" in properties):
                volume = parse_float(line[len("Mesh Volume"):])[-1]
                properties[
                    'mass'] = obj_density * volume / 1000  # The mesh is scaled up by 10 in each dim

            if ("Center of Mass" in line) and (not "com" in properties):
                com = parse_float(line[len("Center of Mass"):])
                properties['com'] = com / 10

            if ("Inertia Tensor" in line) and (not "inertia" in properties):
                inertia_tensor = np.zeros((3, 3))
                inertia_tensor[0, :] = parse_float(log[-idx])
                inertia_tensor[1, :] = parse_float(log[-idx + 1])
                inertia_tensor[2, :] = parse_float(log[-idx + 2])
                properties['inertia'] = inertia_tensor * obj_density / 100000

            if ("mass" in properties) and ("com"
                                           in properties) and ("inertia"
                                                               in properties):
                break
    
    ixx = properties['inertia'][0][0]
    ixy = properties['inertia'][0][1]
    ixz = properties['inertia'][0][2]
    iyy = properties['inertia'][1][1]
    iyz = properties['inertia'][1][2]
    izz = properties['inertia'][2][2]

    # Write the urdf file
    urdf_file_name = urdf_path.split('/')[-1]
    with open(urdf_path, "w+") as f:
        f.write('<?xml version=\"1.0\" ?>\n')
        f.write('<robot name=\"' + urdf_file_name + '\">\n')
        f.write('  <link name=\"baseLink\">\n')
        f.write('    <contact>\n')
        f.write('      <lateral_friction value=\"1.0\"/>\n')
        f.write('      <inertia_scaling value=\"1.0\"/>\n')
        f.write('    </contact>\n')
        f.write('    <inertial>\n')
        f.write(
            '      <origin rpy=\"0 0 0\" xyz=\"%.6f %.6f %.6f\"/>\n' %
            (properties["com"][0], properties["com"][1], properties["com"][2]))
        f.write('      <mass value=\"%.6f\"/>\n' % properties["mass"], )
        f.write(
            '      <inertia ixx=\"%.6f\" ixy=\"%.6f\" ixz=\"%.6f\" iyy=\"%.6f\" iyz=\"%.6f\" izz=\"%.6f\"/>\n'
            % (ixx, ixy, ixz, iyy, iyz, izz))
        f.write('    </inertial>\n')
        f.write('    <visual>\n')
        f.write('      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n')
        f.write('      <geometry>\n')
        f.write('\t\t\t\t<mesh filename=\"' +
                obj_name + '_transform.obj\" scale=\"1 1 1\"/>\n')
        f.write('      </geometry>\n')
        f.write('       <material name=\"white\">\n')
        f.write('        <color rgba=\"1 1 1 1\"/>\n')
        f.write('      </material>\n')
        f.write('    </visual>\n')
        f.write('    <collision>\n')
        f.write('      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n')
        f.write('      <geometry>\n')
        f.write('        <mesh filename=\"' +
                 obj_name + '_transform_vhacd.obj\" scale=\"1 1 1\"/>\n')
        f.write('      </geometry>\n')
        f.write('    </collision>\n')
        f.write('  </link>\n')
        f.write('</robot>\n')
    
    com_pos = np.array([properties["com"][0], properties["com"][1], properties["com"][2]])

    return com_pos

if __name__ == '__main__':
  start_time = time.time()

  data_dir = '/home/xin/Dropbox/SR2021_UC/dataset/real_test_set'
  obj_name_list = os.listdir(data_dir)
  object_input_subdir = "origin"
  object_output_subdir = "transform_20221123"
  for obj_name in obj_name_list:
    print('-----------------------------')
    print(obj_name)
    obj_dir = os.path.join(data_dir, obj_name)
    obj_output_dir = os.path.join(obj_dir, object_output_subdir)
    if not os.path.exists(obj_output_dir):
      os.mkdir(obj_output_dir)
    obj_functional_rotm_annotation_csv = obj_dir + "/origin/" + obj_name + "_annotation.csv"
    agent_sitting_pose_annotation_txt = obj_dir + "/origin/" + obj_name + "_annotation.txt"

    input_file = os.path.join(obj_dir, object_input_subdir, obj_name + '.obj') 
    csv_file = os.path.join(obj_output_dir, obj_name + '_transform.csv')
    obb_transform_file = os.path.join(obj_output_dir, obj_name + '_transform.obj')
    output_file = os.path.join(obj_output_dir, obj_name + '_transform_vhacd.obj')
    urdf_file = os.path.join(obj_output_dir, obj_name + '_transform.urdf')

    obb_transform(input_file, obb_transform_file, csv_file)
    run_vhacd(vhacd_dir, obb_transform_file, output_file)
    meshlabcompute(output_file, urdf_file, obj_name, 600, meshlabserver_exe, meshlab_mlx, meshlab_output_txt)

    # import ipdb; ipdb.set_trace()
  
  process_time = time.time() - start_time
  print("Total process time: ", process_time)