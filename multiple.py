import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
import tracemalloc
from main import simple_analisis

#matplotlib.use('TkAgg')

Text = 20  # Initial exterior temperature
Tint = 500  # Initial interior temperature
L_ = 15  # Length of square
N_ = np.arange(5, 100, 5)  # Mesh size
print(N_)
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

        # Error calc
        ratio = N_[len(N_)-1] // N_[i]
        M_ref_ds = M_ref[::ratio, ::ratio]
        if M_ref_ds.shape != (C.shape):
            M_ref_ds = M_ref_ds[:C.shape[0], :C.shape[1]]
        err = np.linalg.norm(C - M_ref_ds) / np.linalg.norm(M_ref_ds)
        error = np.append(error, err)

    return tt_, mem_, error


def trace_mem_time_err(N, mem, err):
    fig1, ax1 = plt.subplots(1, 3, sharex=True)
    ax1[0].plot(N, mem)
    ax1[1].plot(N, tt)
    ax1[2].plot(N, err)
    plt.show()


tt, mem, err = multiple_analisis(N_, dxy_, Text, Tint)

print(tt.shape, mem.shape, err.shape)
print(tt, mem, err)
trace_mem_time_err(N_, mem, err)

