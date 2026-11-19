import numpy as np

def least_squares_estimation(X1, X2):
	""" 
	inputs:
	X1: Nx3 Matrix Calibrated Point
	X2: Nx3 Matrix Calibrated Point
	X1 ~ X2

	rets:
	E: Essential Matrix using 8 point algorithm
	"""
	##### STUDENT CODE START #####

	A = np.zeros((X1.shape[0], 9))
	for i in range(X1.shape[0]):
		x1 = X1[i, 0]
		y1 = X1[i, 1]
		x2 = X2[i, 0]
		y2 = X2[i, 1]
		A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

	u, s, vh = np.linalg.svd(A)
	E_init = vh[-1].reshape(3, 3)

	# Enforce rank 2 constraint
	u, s, vh = np.linalg.svd(E_init)
	s[2] = 0
	E = u @ np.diag(s) @ vh

	##### STUDENT CODE END #####
	return E
