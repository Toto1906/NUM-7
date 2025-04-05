import numpy as np
import matplotlib.pyplot as plt

T_ext = 20
T_int = 50
L = 1
N = 5
dxy = L/N

A = np.zeros(((N+1)**2, (N+1)**2))
print(A.shape)
print(A)

for i in range(N+1, len(A)-N-1):
    A[i][i] = -4/(dxy**2)
    A[i][i-1] = 1/(dxy**2)
    A[i][i+1] = 1/(dxy**2)
    A[i][i-N-1] = 1/(dxy**2)
    A[i][i+N+1] = 1/(dxy**2)

print(A[10].shape)
print(A[10])