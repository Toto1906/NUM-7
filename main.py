import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('TkAgg')

T_ext = 20      # Initial exterior temperature
T_int = 500     # Initial interior temperature
L = 15          # Length of square
N = 100         # Mesh size
dxy = L/(N+1)

A = np.zeros(((N+1)**2, (N+1)**2))  # Matrix A initialization

B = np.zeros(((N+1)**2))            # Vector B initialization

# Setting initial conditions of B
for i in range(len(B)):
    if i%(N+1) == 0 or (i)%(N+1)==N:
        B[i] = T_ext
    if i % (N + 1) > N / 3 and i % (N + 1) < 2 * N / 3:
        if ((N + 1) ** 2) / 3 < i < 2 * ((N + 1) ** 2) / 3:
            B[i] = T_int

for i in range(N+1):
    B[i] = T_ext

for i in range((N+1)**2 - N-1, (N+1)**2):
    B[i] = T_ext

# Second order Taylor approximation of the diffusion equation
for i in range(N+1, len(A)-N-1):
    A[i][i] = -4/(dxy**2)
    A[i][i-1] = 1/(dxy**2)
    A[i][i+1] = 1/(dxy**2)
    A[i][i-N-1] = 1/(dxy**2)
    A[i][i+N+1] = 1/(dxy**2)

# Initial conditions of A
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

# Verification for A
print(A.shape)
print(A)

# Verification for B
print(B.shape)
print(B)

# Solving the problem
C = np.linalg.solve(A, B)
C = C.reshape((N+1, N+1))
print(C)

# Setting the parameters of the plot
x = np.linspace(0, L, N+1)
y = np.linspace(0, L, N+1)
X, Y = np.meshgrid(x, y)

# Inside rectangle of the plate
rect = patches.Rectangle((L/3, L/3), L/3, L/3, linewidth=1, edgecolor='black', facecolor='none')

# Plotting the temperature
fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, C, cmap='plasma')
ax.add_patch(rect)
ax.set_xticks(ticks=(0, L/3, 2*L/3, L), labels=('0', r'$\frac{L}{3}$', r'$\frac{2L}{3}$', r'$L$'))
ax.set_yticks(ticks=(0, L/3, 2*L/3, L), labels=('0', r'$\frac{L}{3}$', r'$\frac{2L}{3}$', r'$L$'))
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_title(r'Temperature diffusion of a plate of temperatures $T_{ext}=$' + f'{T_ext}' + r' and $T_{int}=$' + f'{T_int}')
ax.set_aspect('equal')
fig.colorbar(mappable=im, label=r'Temperature ($^{\circ}$C)')
fig.savefig('Temperature_diffusion.png', dpi=200)
plt.show()
