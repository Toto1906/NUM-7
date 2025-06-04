import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import tracemalloc
from simple_non_optimized import simple_analisis
import zoom     # For error calculation
import upper_lower_bound_curve_fitting as fit

matplotlib.use('TkAgg')

Text = 20  # Initial exterior temperature
Tint = 500  # Initial interior temperature
L_ = 15  # Length of square
N_ = np.arange(5, 101, 1)  # Mesh size
dxy_ = L_ / (N_ + 1)


def multiple_analisis(N, dxy, T_ext, T_int):
    '''
    :param N: Array of mesh coarseness
    :param dxy: Array of step between each node of the mesh (symmetric in x and y directions)
    :param T_ext: Exterior temperature
    :param T_int: Interior temperature
    '''
    tt_ = np.array([])
    mem_ = np.array([])
    error = np.array([])
    M_ref = np.loadtxt('ref_sol_N=800.txt')

    for i in range(len(N)):
        print(f'\n::: Solving for N = {N[i]}:')
        start = time.time()
        tracemalloc.start()
        C = simple_analisis(N[i], dxy[i], T_ext, T_int)
        current, peak = tracemalloc.get_traced_memory()
        print(f'Peak memory: {peak}')
        tracemalloc.stop()
        peak_mem = peak // 1024
        end = time.time()
        run_time = end - start
        print(f'Time taken: {run_time}\n')
        print('--------------------')
        tt_ = np.append(tt_, run_time)
        mem_ = np.append(mem_, peak_mem)

        # New error calc
        new_lil_matrix = zoom.zoom_lil_matrix(C, M_ref.shape, C.shape)
        err = zoom.error(M_ref, new_lil_matrix)
        error = np.append(error, err)

    return tt_, mem_, error


def trace_mem_time_err(N, mem, err, tt):
    '''
    :param N: Array of on mesh coarseness
    :param mem: Array of peak memory usage for each element in N
    :param err: Array of errors for each element in N
    :param tt: Array of time taken for each element in N
    '''

    # Plot of time and memory usage for each mesh of size N
    fig1, ax1 = plt.subplots(1, 1, sharex=True)
    ax2 = ax1.twinx()
    ax1.plot(N, mem, 's--', label='Memory (kB)', color='gray')
    ax2.plot(N, tt, 'o-', label='Time (s)')
    ax1.set_title('Performance vs N')
    ax1.set_xlabel('N')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0))
    ax2.legend(loc='upper left',bbox_to_anchor=(0.0, 0.0))
    fig1.savefig(f'N={N[-1]}_Performance_vs_N.png')

    # Plot of error vs N
    fig3, ax3 = plt.subplots(1, 1, sharex=True)
    ax3.set_xlabel('N')
    ax3.plot(N, err, marker='o')
    ax3.set_title('Error')
    ax3.set_xlabel('N')
    ax3.set_title('Error vs N')
    ax3.set_xlabel('N')

    fig3.savefig(f'N={N[-1]}_Error_vs_N.png')

    plt.show()


tt, mem, err = multiple_analisis(N_, dxy_, Text, Tint)
trace_mem_time_err(N_, mem, err, tt)
fit.fitted_curves_plot(N_, err, mem, tt)

