import Optimised_functions as f

N = 100
T, x, y, _, _ = f.solve_quarter_plate(N)
f.plot_temperature(T, x, y, L=15)