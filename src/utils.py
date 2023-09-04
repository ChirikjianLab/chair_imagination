# Utils functions


import struct
import math
import numpy as np
import warnings
import pybullet as p


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis_magnitude = np.linalg.norm(axis)
    axis = np.divide(axis,
                     axis_magnitude,
                     out=np.zeros_like(axis),
                     where=axis_magnitude != 0)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.array(np.outer(axis, axis) * (1.0 - cosa))
    axis *= sina
    RA = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    R = RA + np.array(R)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert (isRotm(R))

    if ((abs(R[0][1] - R[1][0]) < epsilon)
            and (abs(R[0][2] - R[2][0]) < epsilon)
            and (abs(R[1][2] - R[2][1]) < epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1] + R[1][0]) < epsilon2)
                and (abs(R[0][2] + R[2][0]) < epsilon2)
                and (abs(R[1][2] + R[2][1]) < epsilon2)
                and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if ((xx > yy) and (xx > zz)):  # R[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz):  # R[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) +
                (R[0][2] - R[2][0]) * (R[0][2] - R[2][0]) +
                (R[1][0] - R[0][1]) * (R[1][0] - R[0][1]))  # used to normalise
    if (abs(s) < 0.001):
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]


def make_rigid_transformation(pos, rotm):
    """
    Rigid transformation from position and orientation.
    Args:
    - pos (3, numpy array): translation
    - rotm (3x3 numpy array): orientation in rotation matrix
    Returns:
    - homo_mat (4x4 numpy array): homogenenous transformation matrix
    """
    homo_mat = np.c_[rotm, np.reshape(pos, (3, 1))]
    homo_mat = np.r_[homo_mat, [[0, 0, 0, 1]]]

    return homo_mat


def rotm2quat(R):
    """
    Rotation matrix to quaternion (x, y, z, w)
    """
    [angle, x, y, z] = rotm2angle(R)
    qw = np.cos(angle / 2)
    qx = x * np.sin(angle / 2)
    qy = y * np.sin(angle / 2)
    qz = z * np.sin(angle / 2)
    return np.array([qx, qy, qz, qw])


def quat2rotm(quat):
    """
    Quaternion to rotation matrix.
    
    Args:
    - quat (4, numpy array): quaternion w, x, y, z
    Returns:
    - rotm: (3x3 numpy array): rotation matrix
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    s = w * w + x * x + y * y + z * z

    rotm = np.array([[
        1 - 2 * (y * y + z * z) / s, 2 * (x * y - z * w) / s,
        2 * (x * z + y * w) / s
    ],
                     [
                         2 * (x * y + z * w) / s, 1 - 2 * (x * x + z * z) / s,
                         2 * (y * z - x * w) / s
                     ],
                     [
                         2 * (x * z - y * w) / s, 2 * (y * z + x * w) / s,
                         1 - 2 * (x * x + y * y) / s
                     ]])

    return rotm


def pose_inv(pose):
    """
    Inverse of a homogenenous transformation.

    Args:
    - pose (4x4 numpy array)

    Return:
    - inv_pose (4x4 numpy array)
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    inv_R = R.T
    inv_t = -np.dot(inv_R, t)

    inv_pose = np.c_[inv_R, np.transpose(inv_t)]
    inv_pose = np.r_[inv_pose, [[0, 0, 0, 1]]]

    return inv_pose


def get_mat_log(R):
    """
    Get the log(R) of the rotation matrix R.
    Args:
    - R (3x3 numpy array): rotation matrix
    Returns:
    - w (3, numpy array): log(R)
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    w_hat = (R - R.T) * theta / (2 * np.sin(theta))  # Skew symmetric matrix
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])  # [w1, w2, w3]

    return w


def sci(n):
    return "{:.6e}".format(n)


def matrixexponential(matrix):
    '''
    Returns the exponential coordinate of the matrix.
    matrix is a (3, 3) np array.
    '''
    if matrix.shape[0] != 3 or matrix.shape[0] != 3:
        raise ValueError("The input rotational matrix is not 3x3!!")

    trace = np.trace(matrix)
    if abs(trace - 1) > 2:
        if abs(trace) < (3 + 0.0001):
            trace = 3
        else:
            print("The matrix trace is too big: {}".format(trace))
    return np.arccos((trace - 1) / 2)  # The range of np.arccos is [0, pi]


def rotx_ea(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


def roty_ea(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def rotz_ea(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a),
                                                  np.cos(a), 0], [0, 0, 1]])


def get_com_pos_obb(obj_urdf):
    """
    Get the com position w.r.t. to the obb frame
    """
    p.connect(p.DIRECT)
    obj_id = p.loadURDF(obj_urdf)
    pos, orn = p.getBasePositionAndOrientation(obj_id)
    p.disconnect()

    return np.array(pos)


def range_angle(theta):
    """
    Limit the range of the angle within [0, 2*pi]
    """
    if theta < 0:
        while theta < 0:
            theta += 2 * np.pi

        return theta
    else:
        while theta > 2 * np.pi:
            theta -= 2 * np.pi

        return theta


def find_nearest_half_pi(theta):
    """
    Try to find the nearest n * pi/2
    theta should be [0, 2 * pi]
    """
    div = np.floor(theta / (np.pi / 2))
    mod = theta - (np.pi / 2) * div

    if (np.pi / 2 - mod) < mod:
        return (div + 1) * np.pi / 2, np.pi / 2 - mod
    else:
        return div * np.pi / 2, mod


def sci(n):
    return "{:.6e}".format(n)

def save_curr_com_pose(chair_urdf, obb_xform_txt, com_pose_txt):
    """Compute current com pose from obb pose and save txt.
    The format for the data is (x, y, z, qw, qx, qy, qz).

    Args:
        chair_urdf (string): chair urdf file
        obb_xform_txt (string): chair obb xform file
        com_pose_txt(string): path to save current com pose
    """
    com_in_obb = np.zeros(3)
    with open(chair_urdf) as f:
        for idx, line in enumerate(f):
            if idx == 8:
                com_pos_list = line.split('"')[3].split()
                for i in range(3):
                    com_in_obb[i] = float(com_pos_list[i])

    obb_xform_rotm = np.zeros((3, 3))
    with open(obb_xform_txt, 'r') as f:
        for idx, line in enumerate(f):
            array = [float(x) for x in line.split()]
            if idx < 3:
                obb_xform_rotm[idx, :] = np.array(array)
            elif idx == 3:
                obb_xform_pos = np.array(array)
    
    chair_curr_com_pos = np.matmul(obb_xform_rotm, com_in_obb) + obb_xform_pos
    # Quaternion in (x, y, z, w)
    chair_curr_com_quat = rotm2quat(obb_xform_rotm)

    with open(com_pose_txt, "w") as f:
        f.write(sci(chair_curr_com_pos[0]) + ",")
        f.write(sci(chair_curr_com_pos[1]) + ",")
        f.write(sci(chair_curr_com_pos[2]) + "\n")
        f.write(sci(chair_curr_com_quat[3]) + ",")
        f.write(sci(chair_curr_com_quat[0]) + ",")
        f.write(sci(chair_curr_com_quat[1]) + ",")
        f.write(sci(chair_curr_com_quat[2]))