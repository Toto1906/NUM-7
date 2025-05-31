import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import tracemalloc
from main import simple_analisis
import zoom
import upper_lower_bound_curve_fitting as fit

matplotlib.use('TkAgg')

Text = 20  # Initial exterior temperature
Tint = 500  # Initial interior temperature
L_ = 15  # Length of square
N_ = np.arange(5, 100, 5)  # Mesh size
dxy_ = L_ / (N_ + 1)


def multiple_analisis(N, dxy, T_ext, T_int):
    tt_ = np.array([])
    mem_ = np.array([])
    error = np.array([])
    M_ref = simple_analisis(N[len(N)-1], dxy[len(dxy)-1], T_ext, T_int)

    for i in range(len(N)):
        start = time.time()
        tracemalloc.start()
        C = simple_analisis(N[i], dxy[i], T_ext, T_int)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem = peak // 1024
        end = time.time()
        run_time = end - start
        tt_ = np.append(tt_, run_time)
        mem_ = np.append(mem_, peak_mem)

        # New error calc
        new_lil_matrix = zoom.zoom_lil_matrix(C, M_ref.shape, C.shape)
        err = zoom.error(M_ref, new_lil_matrix)
        error = np.append(error, err)

    return tt_, mem_, error


def trace_mem_time_err(N, mem, err):
    fig1, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(N, mem, 's--', label='Memory (kB)', color='gray')
    ax1.set_title('Memory required')
    ax1.set_xlabel('N')
    ax1.plot(N, tt, 'o-', label='Time (s)')
    ax1.set_title('Time required')
    ax1.grid(True, which='both', linestyle='--')

    fig2, ax2 = plt.subplots(1, 1, sharex=True)
    ax2.set_xlabel('N')
    ax2.plot(N, err)
    ax2.set_title('Error')
    ax2.set_xlabel('N')
    plt.show()


tt, mem, err = multiple_analisis(N_, dxy_, Text, Tint)
trace_mem_time_err(N_, mem, err)
# fit.fitted_curves_plot(N_, err)

