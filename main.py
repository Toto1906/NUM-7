import numpy as np
import matplotlib.pyplot as plt

T_ext = 20
T_int = 50
L = 1
N = 5
dxy = L/N

A = np.zeros((N+1, N+1, N+1))

for i in range(1, N-1):
    for j in range(1, N-1):
        A[i][j] = -4/(dxy**2)
        A[i][j-1] = 1/(dxy**2)
        A[i][j+1] = 1/(dxy**2)
        A[i][j] = 1/(dxy**2)
        A[i][j] = 1/(dxy**2)

print(A)