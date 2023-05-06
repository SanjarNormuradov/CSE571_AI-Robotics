import numpy as np
import matplotlib.pyplot as plt
import math

def minimized_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle >= np.pi:
        angle -= 2 * np.pi
    return angle

def rotation_matrix_from_rpy(roll, pitch, yaw):
    """Generalized rotation matrix from roll, pitch, yaw (degrees)"""
    roll_rad, pitch_rad, yaw_rad = np.radians(roll), np.radians(pitch), np.radians(yaw)
    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)

    R_roll = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    R_pitch = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    R_yaw = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    R = R_yaw @ R_pitch @ R_roll
    return R
