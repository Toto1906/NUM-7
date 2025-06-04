# -*- coding: utf-8 -*-
"""
Created on Mon May 12 22:34:43 2025

@author: remyk
"""
import Optimised_functions as f
import time
import numpy as np
# test^2
#monitor time of whole operation
initial_time = time.time()

N_max = 200
coarser = np.arange(5, N_max, 1)

results = f.mesh_error_vs_reference(coarser)

# Plot reference solution
Tmax, xmax, ymax, _, _ = f.solve_quarter_plate(N_max)
#monitor time of whole operation
operation_time = time.time() - initial_time
print(f"Operation time : {operation_time}")

f.plot_temperature(Tmax, xmax, ymax, L=15)



# Plot error convergence and performance
f.plot_convergence(results, coarser)

import upper_lower_bound_curve_fitting
upper_lower_bound_curve_fitting.fitted_curves_plot(results['n'], results['error'], results['memory'], results['time'])
