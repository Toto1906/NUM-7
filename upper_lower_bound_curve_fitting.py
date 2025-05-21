
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

# Define model functions
def fitted_curves_plot(N_values_vec, error_values_vec):

    def power_model(N, a, b):
        return a * N ** (-b)

    # Extract data

    N_vals = np.array(N_values_vec)
    errors = np.array(error_values_vec)

    peaks_pos, _ = find_peaks(errors, distance=5)
    print(peaks_pos)
    peaks_neg, _ = find_peaks(-errors, distance=5)
    print(peaks_neg)

    # Upper Power law fit
    popt_pow_high, _ = curve_fit(power_model, N_vals[peaks_pos], errors[peaks_pos], p0=(1.0, 1.0))
    pow_fit_high = power_model(N_vals[peaks_pos], *popt_pow_high)
    r2_pow_high = r2_score(errors[peaks_pos], pow_fit_high)

    # Lower Power law fit
    popt_pow_low, _ = curve_fit(power_model, N_vals[peaks_neg], errors[peaks_neg], p0=(1.0, 1.0))
    pow_fit_low = power_model(N_vals[peaks_neg], *popt_pow_low)
    r2_pow_low = r2_score(errors[peaks_neg], pow_fit_low)

    # Print fitted models and R²
    print(f"Upper Power law fit : error ≈ {popt_pow_high[0]:.2e} · N^(-{popt_pow_high[1]:.3f}), R² = {r2_pow_high:.4f}")
    print(f"Lower Power law fit : error ≈ {popt_pow_low[0]:.2e} · N^(-{popt_pow_low[1]:.3f}), R² = {r2_pow_low:.4f}")

    # Smooth N values for plotting
    N_smooth = np.linspace(min(N_vals), max(N_vals), 300)
    pow_smooth_high = power_model(N_smooth, *popt_pow_high)
    pow_smooth_low = power_model(N_smooth, *popt_pow_low)

    # Plot in linear scale
    plt.figure()
    plt.plot(N_vals, errors, 'ko', label='Data')
    plt.plot(N_smooth, pow_smooth_high, 'r-', label=f'Upper Power law fit (R²={r2_pow_high:.4f})')
    plt.plot(N_smooth, pow_smooth_low, 'b--', label=f'Lower Power law fit (R²={r2_pow_low:.4f})')
    plt.xlabel('N')
    plt.ylabel('Relative error')
    plt.title('Error vs N (Linear Scale)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    # Plot in log-log scale
    plt.figure()
    plt.loglog(N_vals, errors, 'ko', label='Data')
    plt.loglog(N_smooth, pow_smooth_high, 'r-', label='Upper Power law fit')
    plt.loglog(N_smooth, pow_smooth_low, 'b--', label='Lower Power law fit')
    plt.xlabel('N (log scale)')
    plt.ylabel('Error (log scale)')
    plt.title('Log-Log Plot: Error vs N')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()