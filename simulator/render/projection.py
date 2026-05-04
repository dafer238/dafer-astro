import numpy as np


def perspective_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    mat = np.zeros((4, 4))
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def project_points(points_3d: np.ndarray, view_mat: np.ndarray,
                   proj_mat: np.ndarray, width: float, height: float) -> np.ndarray:
    n = len(points_3d)
    if n == 0:
        return np.zeros((0, 2))

    homo = np.ones((n, 4))
    homo[:, :3] = points_3d

    clip = (proj_mat @ view_mat @ homo.T).T

    w = clip[:, 3]
    valid = w > 0.01
    ndc = np.zeros((n, 2))
    ndc[valid, 0] = clip[valid, 0] / w[valid]
    ndc[valid, 1] = clip[valid, 1] / w[valid]

    screen = np.zeros((n, 2))
    screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * width
    screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * height

    screen[~valid] = -9999.0
    return screen
