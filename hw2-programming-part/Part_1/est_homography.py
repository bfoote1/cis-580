import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    A = np.zeros((8, 9))

    for i in range(4):
        x_i, y_i = X[i]
        u_i, v_i = Y[i]

        A[2*i] = [x_i, y_i, 1, 0, 0, 0,
                -u_i*x_i, -u_i*y_i, -u_i]

        A[2*i + 1] = [0, 0, 0, x_i, y_i, 1,
                    -v_i*x_i, -v_i*y_i, -v_i]

    U, S, Vt = np.linalg.svd(A)

    h = Vt[-1]
    H = h.reshape(3, 3)
    ##### STUDENT CODE END ##### 

    return H


if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0],[0, 10], [5, 0], [5, 10]])
    Y = np.array([[3, 4], [4, 11],[8, 5], [9, 12]])
    H = est_homography(X, Y)
    print(H)
    