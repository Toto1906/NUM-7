import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
#matplotlib.use('TkAgg')

start_time = time.time()

Text = 20      # Initial exterior temperature
Tint = 500     # Initial interior temperature
L_ = 15          # Length of square
N_ = 10          # Mesh size
dxy_ = L_/(N_+1)

def simple_analisis(N, dxy, T_ext, T_int, multiple=False):
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

    # Solving the problem
    C = np.linalg.solve(A, B)
    C = C.reshape((N+1, N+1))

    return C


# C = simple_analisis(N_, dxy_, Text, Tint)
#
# print("--- %s seconds ---" % (time.time() - start_time))
#
# # Setting the parameters of the plot
# x = np.linspace(0, L_, N_+1)
# y = np.linspace(0, L_, N_+1)
# X, Y = np.meshgrid(x, y)
#
# # Inside rectangle of the plate
# rect = patches.Rectangle((L_/3, L_/3), L_/3, L_/3, linewidth=1, edgecolor='black', facecolor='none')
#
# color='afmhot'
#
# # Plotting the temperature
# fig1, (ax1) = plt.subplots(1, 1)
#
# im = ax1.pcolormesh(X, Y, C, cmap=color)
# ax1.add_patch(rect)
# ax1.set_xticks(ticks=(0, L_ / 3, 2 * L_ / 3, L_), labels=('0', r'$\frac{L}{3}$', r'$\frac{2L}{3}$', r'$L$'))
# ax1.set_yticks(ticks=(0, L_ / 3, 2 * L_ / 3, L_), labels=('0', r'$\frac{L}{3}$', r'$\frac{2L}{3}$', r'$L$'))
# ax1.tick_params(axis='both', which='major', labelsize=10)
# ax1.set_title(r'Temperature diffusion of a plate of temperatures $T_{ext}=$' + f'{Text}' + r' and $T_{int}=$' + f'{Tint}')
# ax1.set_aspect('equal')
# fig1.colorbar(mappable=im, label=r'Temperature ($^{\circ}$C)')
# fig1.savefig('Temperature_diffusion.png', dpi=200)
#
# # ax2.plot(x, C[int(L/2)])
#
# fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y, C, cmap=color)
#
# plt.show()
