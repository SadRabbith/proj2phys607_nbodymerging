import numpy as np
import matplotlib.pyplot as plt

# Example parameters
v0 = 0.5
N = 50000

# Your rejection sampling function
def rejection_sampling_MB(scale, size=1, vmax=None, random_state=None):
    rng = np.random.default_rng(random_state)
    if vmax is None:
        vmax = 6 * scale

    def pdf(v):
        return v**2 * np.exp(-(v/scale)**2)

    v_peak = scale * np.sqrt(2/3)
    p_max = pdf(v_peak)

    samples = []
    while len(samples) < size:
        v_candidate = rng.uniform(0, vmax)
        u = rng.random()
        if u < pdf(v_candidate) / p_max:
            samples.append(v_candidate)
    return np.array(samples)

# Generate velocities
velocities = rejection_sampling_MB(scale=v0, size=N)

# Analytic Maxwell–Boltzmann PDF for comparison
v = np.linspace(0, 3*v0, 200)
analytic_pdf = (v**2 / v0**3) * np.exp(-(v/v0)**2)
analytic_pdf /= np.trapz(analytic_pdf, v)  # normalize

# Plot
plt.figure(figsize=(6,4))
plt.hist(velocities, bins=60, density=True, alpha=0.6, label='Sampled (Rejection Sampling)')
plt.plot(v, analytic_pdf, 'r--', lw=2, label='Analytic Maxwell–Boltzmann')
plt.xlabel('Speed v')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Velocity Distribution by Rejection Sampling')
plt.tight_layout()
plt.show()
