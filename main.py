import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

T_ext = 20
T_int = 50
L = 15
N = 100
dxy = L/(N+1)

A = np.zeros(((N+1)**2, (N+1)**2))

B = np.ones(((N+1)**2))

for i in range(len(B)):
    if i%(N+1) == 0 or (i)%(N+1)==N:
        B[i] = T_ext
    if N/3 <= i%(N+1) <= (2*N)/3:
        B[i] = T_int

for i in range(N+1):
    B[i] = T_ext

for i in range((N+1)**2 - N-1, (N+1)**2):
    B[i] = T_ext

for i in range(N+1, len(A)-N-1):
    A[i][i] = -4/(dxy**2)
    A[i][i-1] = 1/(dxy**2)
    A[i][i+1] = 1/(dxy**2)
    A[i][i-N-1] = 1/(dxy**2)
    A[i][i+N+1] = 1/(dxy**2)

for i in range(N+1, len(A)-N-1):
    if i % (N + 1) > N/3 and i % (N + 1) < 2*N/3:
        if ((N+1)**2)/3 < i < 2*((N+1)**2)/3:
            A[i] = np.zeros((N + 1) ** 2)
            A[i][i] = 1
    if i % (N + 1) == 0 or (i) % (N + 1) == N:
        A[i] = np.zeros((N + 1) ** 2)
        A[i][i] = 1



for i in range(N+1):
    A[i][i] = 1

for i in range((N+1)**2 - N-1, (N+1)**2):
    A[i][i] = 1

print(A.shape)
print(A)



print(B.shape)
print(B)

C = np.linalg.solve(A,B)

C = C.reshape((N+1, N+1))
print(C)

x = np.linspace(0 , L, N+1)
y = np.linspace(0 , L, N+1)

X, Y = np.meshgrid(x, y)

plt.pcolormesh(X, Y, C, cmap='plasma')
plt.colorbar()
plt.show()