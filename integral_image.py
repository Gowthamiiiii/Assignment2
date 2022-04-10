import numpy as np
import cv2

img = cv2.imread('i1.jpg')
# matrix_A = [[1,0,0,1,0],[0,0,0,1,0],[0,1,0,1,0],[1,1,0,1,0],[1,0,0,0,1]]
A = np.array(img)
m,n,n_channels = A.shape
for i in range(m):
    initial_sum = 0
    for j in range(n):
        initial_sum += A[i][j]
        A[i][j] = initial_sum
        if i > 0:
            A[i][j] += A[i-1][j]
print(A)
