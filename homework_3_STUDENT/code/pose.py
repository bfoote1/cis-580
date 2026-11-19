import numpy as np

def pose_candidates_from_E(E):
    """ 
    Computes four possible camera poses (rotation and translation) from the essential matrix.

    Inputs:
    - E: 3x3 Matrix, Essential Matrix

    Outputs:
    - transform_candidates: List of 4 dictionaries, each containing:
        - "T": 3x1 Vector, Translation vector
        - "R": 3x3 Matrix, Rotation matrix
    """
    
    ##### STUDENT CODE START #####
    U, _, Vt = np.linalg.svd(E)
    
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=float)
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    T = U[:, 2]
    transform_candidates = []
    
    for R in [R1, R2]:
        if np.linalg.det(R) < 0:
            R = -R
        for T_sign in [T, -T]:
            transform_candidates.append({
                "R": R.copy(),
                "T": T_sign.copy()
            })
    
    ##### STUDENT CODE END #####
    
    return transform_candidates