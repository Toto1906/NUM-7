import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import tracemalloc
import zoom

matplotlib.use('TkAgg')

T_reference_file = np.loadtxt('ref_sol_N=800.txt')

def solve_quarter_plate(N, L=15, T_ext=20, T_int=500):
    """
    Solve one quarter of the square plate using symmetry+Neumann BCs,
    reconstruct full plate, return T_full, x_full, y_full, elapsed time, peak memory.
    """
    dxy = L / (N + 1)  # spacing between grid points
    Nq = (N + 1) // 2  # number of points on one axis in the quarter (excluding symmetry)
    Nx = Ny = Nq + 1  # include symmetry edge → number of nodes in x and y

    start = time.time()  # start monitoring the time
    tracemalloc.start()  # start monitoring the memory usage

    A = lil_matrix((Nx * Ny,
                    Nx * Ny))  # initialise A matrix, we use lil_matrix since it is easy to construct incremental matrices with and more importantly stores sparse matrices very efficently
    B = np.zeros(Nx * Ny)  # initialise B vector

    def idx(i, j):
        "Used to flatten the 2D coordinates into a 1D index"
        return i * Nx + j

    for i in range(Ny):
        for j in range(Nx):
            k = idx(i, j)
            x = L / 2 + j * dxy
            y = L / 2 + i * dxy

            # hole region
            if L / 3 <= x <= 2 * L / 3 and L / 3 <= y <= 2 * L / 3:
                A[k, k] = 1;
                B[k] = T_int;
                continue

            # outer Dirichlet boundaries
            if i == Ny - 1 or j == Nx - 1:
                A[k, k] = 1;
                B[k] = T_ext;
                continue

            # bottom symmetry edge: Neumann (zero flux accross symmetry axis)
            if i == 0:
                A[k, k] = -4 / dxy ** 2
                A[k, idx(i + 1, j)] = 2 / dxy ** 2
                if j > 0:   A[k, idx(i, j - 1)] = 1 / dxy ** 2
                if j < Nx - 1: A[k, idx(i, j + 1)] = 1 / dxy ** 2
                B[k] = 0;
                continue

            # left symmetry edge: Neumann (zero flux accross symmetry axis)
            if j == 0:
                A[k, k] = -4 / dxy ** 2
                A[k, idx(i, j + 1)] = 2 / dxy ** 2
                if i > 0:   A[k, idx(i - 1, j)] = 1 / dxy ** 2
                if i < Ny - 1: A[k, idx(i + 1, j)] = 1 / dxy ** 2
                B[k] = 0;
                continue

            # interior points
            A[k, k] = -4 / dxy ** 2
            A[k, idx(i + 1, j)] = 1 / dxy ** 2
            A[k, idx(i - 1, j)] = 1 / dxy ** 2
            A[k, idx(i, j + 1)] = 1 / dxy ** 2
            A[k, idx(i, j - 1)] = 1 / dxy ** 2
            B[k] = 0

    A_csr = csr_matrix(
        A)  # converts the matrix to Compressed Sparse Row format to same computation time, spsolve is very efficient
    Tq = spsolve(A_csr, B).reshape((Ny, Nx))  # Solves the system efficently and reshapes it back into a matrix form

    current, peak = tracemalloc.get_traced_memory()  #
    tracemalloc.stop()
    elapsed = time.time() - start
    peak_mem = peak // 1024

    # mirror to full plate
    Q1 = Tq
    Q2 = np.fliplr(Q1[:, 1:])
    Q3 = np.flipud(Q1[1:, :])
    Q4 = np.flipud(np.fliplr(Q1[1:, 1:]))
    T_full = np.block([[Q4, Q3], [Q2, Q1]])

    x_full = np.linspace(0, L, N + 1)
    y_full = np.linspace(0, L, N + 1)
    return T_full, x_full, y_full, elapsed, peak_mem


def mesh_error_vs_reference(N_values, T_ref= T_reference_file, L=15):
    """
    Compute reference solution at N_ref, then for each N in N_values,
    compute solution and error relative to reference (downsampled).

    Returns dict:
      'h', 'error', 'time', 'memory'.
    """
    results = {'n': [], 'error': [], 'time': [], 'memory': []}

    for i in range(len(N_values)):
        T, x, y, t, m = solve_quarter_plate(N_values[i], L)
        # ratio = (N_ref + 1) // (N + 1)
        # T_ref_ds = T_ref[::ratio, ::ratio]
        # if T_ref_ds.shape != T.shape:
        #     T_ref_ds = T_ref_ds[:T.shape[0], :T.shape[1]]
        # err = np.linalg.norm(T - T_ref_ds) / np.linalg.norm(T_ref_ds)

        new_err_matrix = zoom.zoom_lil_matrix(T, T_ref.shape, T.shape)
        err = zoom.error(T_ref, new_err_matrix)

        # new_ref_matrix = zoom.dezoom_ref_matrix(T_ref, T_ref.shape, T.shape)
        # err = zoom.error(new_ref_matrix, T)

        results['n'].append((N_values[i]))
        results['error'].append(err)
        results['time'].append(t)
        results['memory'].append(m)

    return results


def plot_temperature(T, x, y, L):
    X, Y = np.meshgrid(x, y)
    fig1, ax1 = plt.subplots()
    im = ax1.pcolormesh(X, Y, T, shading='auto', cmap='afmhot')
    ax1.add_patch(plt.Rectangle((L / 3, L / 3), L / 3, L / 3, edgecolor='white', facecolor='none'))
    ax1.set_xticks(ticks=(0, L / 3, 2 * L / 3, L), labels=('0', r'$\frac{L}{3}$', r'$\frac{2L}{3}$', r'$L$'))
    ax1.set_yticks(ticks=(0, L / 3, 2 * L / 3, L), labels=('0', r'$\frac{L}{3}$', r'$\frac{2L}{3}$', r'$L$'))
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_title(f'Plate Temperature Distribution N = {len(T)-1}')
    ax1.set_aspect('equal')
    fig1.colorbar(mappable=im, label=r'Temperature ($^{\circ}$C)')

    plt.show()


def plot_convergence(results, N_values):
    """
    Plot error vs grid spacing using reference-based errors, and performance vs N.
    """
    #     # Sort h and corresponding errors ascending by h
    #     pairs = sorted(zip(results['n'], results['error']), key=lambda x: x[0])
    #     h_sorted, err_sorted = zip(*pairs)
    h_sorted, err_sorted = results['n'], results['error']

    plt.figure()
    plt.plot(h_sorted, err_sorted, 'o-')
    plt.xlabel('N')
    plt.ylabel('Relative error vs ref')
    plt.title('Error vs N')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()

    # Performance vs N_values
    times = results['time']
    mems = results['memory']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(N_values, times, 'o-', label='Time (s)')
    ax2.plot(N_values, mems, 's--', label='Memory (kB)', color='gray')
    ax1.set_xlabel('N')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('Peak Memory (kB)')
    fig.legend(loc='upper right')
    plt.title('Performance vs N')
    plt.tight_layout()
    plt.show()
