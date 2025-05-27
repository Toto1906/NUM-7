import Optimised_functions
import numpy as np
N_ref = 800
Ref_sol = Optimised_functions.solve_quarter_plate(N_ref)
Optimised_functions.plot_temperature(Ref_sol[0], Ref_sol[1], Ref_sol[2], L=15)
Ref_sol_T = Ref_sol[0]
print(Ref_sol_T)

np.savetxt(f'ref_sol_N={N_ref}.txt', Ref_sol_T)
