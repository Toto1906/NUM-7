import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import tracemalloc
from main import simple_analisis
import zoom

matplotlib.use('TkAgg')

Text = 20  # Initial exterior temperature
Tint = 500  # Initial interior temperature
L_ = 15  # Length of square
N_ = np.arange(10, 100, 10)  # Mesh size
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
    fig1, ax1 = plt.subplots(1, 3, sharex=True)
    ax1[0].plot(N, mem)
    ax1[0].set_title('Memory required')
    ax1[2].set_xlabel('N')
    ax1[1].plot(N, tt)
    ax1[1].set_title('Time required')
    ax1[2].set_xlabel('N')
    ax1[2].plot(N, err)
    ax1[2].set_title('Error')
    ax1[2].set_xlabel('N')
    plt.show()


tt, mem, err = multiple_analisis(N_, dxy_, Text, Tint)
trace_mem_time_err(N_, mem, err)

