"""URDF utility functions for forward kinematics and joint info."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def get_joint_limits(urdf) -> tuple[np.ndarray, np.ndarray]:
    """
    Get joint limits from a yourdfpy URDF.

    Args:
        urdf: yourdfpy URDF object

    Returns:
        Tuple of (lower_limits, upper_limits) as numpy arrays
    """
    lower = []
    upper = []
    for jname in urdf.actuated_joint_names:
        joint = urdf.joint_map[jname]
        lower.append(joint.limit.lower if joint.limit else -np.pi)
        upper.append(joint.limit.upper if joint.limit else np.pi)
    return np.array(lower), np.array(upper)


def get_link_transforms(urdf, joint_cfg: np.ndarray | None = None) -> np.ndarray:
    """
    Compute forward kinematics for all links using yourdfpy.

    Args:
        urdf: yourdfpy URDF object
        joint_cfg: Joint configuration array. If None, uses middle of joint limits.

    Returns:
        (num_links, 7) array where each row is [qw, qx, qy, qz, x, y, z]
        (quaternion wxyz format + translation)
    """
    if joint_cfg is None:
        lower, upper = get_joint_limits(urdf)
        joint_cfg = (lower + upper) / 2

    # Update URDF configuration (does FK internally)
    urdf.update_cfg(joint_cfg)

    # Get transforms for all links
    link_names = list(urdf.link_map.keys())
    transforms = np.zeros((len(link_names), 7))

    for i, link_name in enumerate(link_names):
        # Get 4x4 homogeneous transform
        T = urdf.get_transform(link_name)

        # Extract rotation matrix and convert to quaternion (wxyz format)
        rot_matrix = T[:3, :3]
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()  # scipy uses xyzw
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Extract translation
        translation = T[:3, 3]

        transforms[i, :4] = quat_wxyz
        transforms[i, 4:] = translation

    return transforms


def get_link_names(urdf) -> list[str]:
    """Get ordered list of link names from URDF."""
    return list(urdf.link_map.keys())


def get_num_actuated_joints(urdf) -> int:
    """Get the number of actuated joints in the URDF."""
    return len(urdf.actuated_joint_names)
