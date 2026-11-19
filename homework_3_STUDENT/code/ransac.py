from lse import least_squares_estimation
import numpy as np
# import tqdm
# import scipy.spatial as ss



def ransac_estimator(X1, X2, num_iterations=60000):
    """
    Estimates the essential matrix using RANSAC algorithm.

    Inputs:
    - X1: Nx3 Matrix, Calibrated points from first camera
    - X2: Nx3 Matrix, Calibrated points from second camera
    - num_iterations: int, Number of RANSAC iterations (default: 60000)

    Outputs:
    - E: 3x3 Matrix, Best essential matrix found
    - inliers: ndarray of shape (n,), Indices of inlier matches
    """
    sample_size = 8

    eps = 1e-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """
        E: ndarray of shape (3,3)
            Essential matrix.
        inliers: ndarray of shape (n,)
            Indices of inlier matches.
        """
        ##### STUDENT CODE START #####

        E = least_squares_estimation(
            X1[sample_indices],
            X2[sample_indices]
        )

        # Epipolar lines in camera 2 and camera 1
        lines2 = (E @ X1.T).T
        lines1 = (E.T @ X2.T).T

        # Algebraic epipolar error
        errors = np.sum(X2 * lines2, axis=1) ** 2

        # Squared norms of the first two line coefficients
        denom2 = np.sum(lines2[:, :2] ** 2, axis=1)
        denom1 = np.sum(lines1[:, :2] ** 2, axis=1)

        # Symmetric squared epipolar distance
        residuals = errors * (
            1 / np.maximum(denom2, 1e-12)
            + 1 / np.maximum(denom1, 1e-12)
        )

        inliers = np.where(residuals < eps)[0]
        
        ##### STUDENT CODE END #####

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers

if __name__ == "__main__":
    T = np.array([0.9, 2, 1.5])
    Tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    Rot = ss.transform.Rotation.from_euler('xyz', [0.5, 0.3, 0.2]).as_matrix()
    E = Tx.dot(Rot)
    q = np.random.rand(9, 3)
    q[:, 0] /= q[:, 2]
    q[:, 1] /= q[:, 2]
    q[:, 2] /= q[:, 2]
    L = q.dot(E)
    p = np.random.rand(9, 3)
    p[:, 0] /= p[:, 2]
    p[:, 1] /= p[:, 2]
    p[:, 2] /= p[:, 2]
    p[:, 0] = (-L[:, 2]-L[:, 1]*p[:, 1])/L[:, 0]
    ransac_estimator(p, q)
