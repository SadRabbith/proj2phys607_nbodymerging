import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import statistics 
import math 
import time 
import sys

G = 1.0 

#2nd order differential equation, m r ddot = sum (G mi mj / |rj - ri|**3 ) * (rj - ri)

class SamplingTool: 

    """ We use Sampeter power law for masses in inverse CDF and Maxwell Boltzmann speeds for rejection sampling 
    """

    @staticmethod
    def inverse_cdf_powerlaw(alpha, xmin, xmax, size = 1, random_state = None): 

        """
        Sample from p(x) ~ x^{-alpha}
        """

        range_val = np.random.default_rng(random_state)
        u = range_val.random(size)

        if alpha == 1.0: 
            return np.exp(u * (np.log(xmax) - np.log(xmin)) + np.log(xmin))
        
        powl = 1.0 - alpha
        min_val = xmin ** powl 
        max_val = xmax ** powl 

        x = (u * (max_val - min_val) +min_val)**(1 / powl) 

        return x 
    
    @staticmethod
    
    def rejection_sampling_MB(scale, size=1, vmax=None, random_state=None):
        rng = np.random.default_rng(random_state)
        if vmax is None:
            vmax = 6 * scale

    # Normalization constant is irrelevant for rejection ratio
        def pdf(v): 
            return v**2 * np.exp(-(v/scale)**2)

    # Find approximate maximum of PDF
        v_peak = scale * np.sqrt(2/3)
        p_max = pdf(v_peak)

        samples = []
        while len(samples) < size:
            v_candidate = rng.uniform(0, vmax)
            u = rng.random()
            if u < pdf(v_candidate) / p_max:
                samples.append(v_candidate)

        return np.array(samples)
    
class StarClusters: 
    """stores star properties""" 

    def __init__(self, masses, positions, velocities):

        assert masses.shape[0] == positions.shape[0] == velocities.shape[0] 

        self.m = masses.astype(float) 
        self.r = positions.astype(float) 
        self.v = velocities.astype(float) 
        self.N = self.m.size 
    
    @classmethod
    def initialize_random(cls, N, mass_alpha = 2.35, mass_min = 0.5, mass_max = 5.0, radius_scale = 1.0, 
                          velocity_scale = 0.5, random_state = None): #MC methods to find masses radial positions and velocities
        
        range_val = np.random.default_rng(random_state)
        masses = SamplingTool.inverse_cdf_powerlaw(alpha = mass_alpha, xmin = mass_min, xmax = mass_max, size = N, random_state=range_val)

        u = range_val.random(size = N) 
        pos = range_val.normal(scale=radius_scale, size = (N,3)) 

        pos = pos / np.linalg.norm(pos, axis = 1)[:, None] *(np.abs(range_val.normal(radius_scale, size =N))[:, None]) 

        pos = range_val.normal(scale = radius_scale, size= (N,3)) 

        speeds = SamplingTool.rejection_sampling_MB(scale = velocity_scale, size = N, random_state= range_val) 

        theta = np.arccos(1 - 2 *range_val.random(size = N)) 
        phi = 2 * np.pi * range_val. random(size = N) 
        vx = speeds * np.sin(theta) * np.cos(phi) 
        vy = speeds* np.sin(theta) * np.sin(phi) 
        vz = speeds * np.cos(theta) 

        vel = np.vstack([vx, vy, vz]).T

        return cls(masses = masses, positions = pos, velocities = vel)


    def get_state_vector(self):
        """Return flattened state vector y = [r.flatten(), v.flatten()]"""
        y = np.concatenate([self.r.ravel(), self.v.ravel()])
        return y


    def update_state_vector(self,y):

        N = self.N 

        r = y[:3*N].reshape((N,3))
        v = y[3*N:].reshape((N,3)) 
        self.r, self.v = r.copy(), v.copy() 

    def remove_index(self, idx): 

        mask = np.ones(self.N, dtype = bool) 
        mask[idx] = False
        self.m = self.m[mask]
        self.r = self.r[mask]
        self.v = self.v[mask]
        self.N = self.m.size


    def merge_pair(self, i, j):
        """Merge stars i and j -> conserve momentum. New star replaces index i; remove j."""

        if j < 0 or j >= self.N:
            return
        m1, m2 = self.m[i], self.m[j]
        total_m = m1 + m2
        new_v = (m1 * self.v[i] + m2 * self.v[j]) / total_m
        # Place merged star at mass-weighted center
        new_r = (m1 * self.r[i] + m2 * self.r[j]) / total_m
        self.m[i] = total_m
        self.v[i] = new_v
        self.r[i] = new_r
        # remove j (ensure we remove the higher index first for safety)
        self.remove_index(j if j > i else j)

    def scatter_pair_elastic(self, i, j):
        """
        Perform elastic scattering between stars i and j.
        
        Uses standard elastic collision formula along line of centers.
        Conserves both momentum and kinetic energy.
        
        Parameters
        ----------
        i, j : int
            Indices of colliding stars
        """
        m1, m2 = self.m[i], self.m[j]
        v1, v2 = self.v[i].copy(), self.v[j].copy()
        
        # Line of centers (collision axis)
        r_rel = self.r[i] - self.r[j]
        r_hat = r_rel / (np.linalg.norm(r_rel) + 1e-12)
        
        # Velocity components along collision axis
        v1_n = np.dot(v1, r_hat)
        v2_n = np.dot(v2, r_hat)
        
        # Perpendicular components (unchanged in collision)
        v1_perp = v1 - v1_n * r_hat
        v2_perp = v2 - v2_n * r_hat
        
        # Standard elastic collision formula for normal components
        v1_n_new = ((m1 - m2) * v1_n + 2 * m2 * v2_n) / (m1 + m2)
        v2_n_new = ((m2 - m1) * v2_n + 2 * m1 * v1_n) / (m1 + m2)
        
        # Reconstruct velocities
        self.v[i] = v1_n_new * r_hat + v1_perp
        self.v[j] = v2_n_new * r_hat + v2_perp

def ode_func(t, y, masses):
    """
    y: flattened [r (3N), v (3N)]
    returns y' = [v, a]
    masses: array of shape (N,)
    """
    N = masses.size
    r = y[:3 * N].reshape((N, 3))
    v = y[3 * N:].reshape((N, 3))
    a = np.zeros_like(r)
    
    for i in range(N):
       
        rij = r - r[i]
        dist3 = np.sum(rij * rij, axis=1) ** (1.5)
        # avoid self division
        dist3[i] = np.inf
        
        a[i] = G * np.sum((masses[:, None] * rij) / dist3[:, None], axis=0)
    dydt = np.concatenate([v.ravel(), a.ravel()])
    return dydt

def default_merge(star_cluster: StarClusters, i, j, v_rel_threshold = 15): 

    v_rel = np.linalg.norm(star_cluster.v[i] - star_cluster.v[j])

    k = 5.0 
    x = (v_rel_threshold - v_rel) / v_rel_threshold

    p = 1.0/ (1.0 + np.exp(-k*x)) 
    
    return float(np.clip(p, 0.0, 1.0))

class Simulator:
    """runs the simulation loop"""

    def __init__(self, cluster: StarClusters, t_end=1.0, r_c=0.1, dt_event=0.01,
                 p_merge_func=None, max_step=0.01, rng_seed=None):
        self.cluster = cluster
        self.t_end = t_end
        self.r_c = r_c
        self.dt_event = dt_event  # interval between event checks
        self.p_merge_func = p_merge_func if p_merge_func is not None else default_merge
        self.max_step = max_step
        self.rng = np.random.default_rng(rng_seed)

        self.energy_history = [] 

    def compute_energy(self):
        """
        Compute total energy (kinetic + potential) of the cluster.
        
        Returns
        -------
        KE : float
            Total kinetic energy
        PE : float
            Total gravitational potential energy
        E_tot : float
            Total energy (KE + PE)
        
        Notes
        -----
        Uses pairwise summation for potential energy calculation.
        Includes small softening parameter (1e-10) to avoid singularities.
        """
        cluster = self.cluster
        
        # Kinetic energy: (1/2) * sum(m * v^2)
        KE = 0.5 * np.sum(cluster.m * np.sum(cluster.v**2, axis=1))
        
        # Potential energy: -G * sum_i sum_{j>i} (mi * mj / rij)
        PE = 0.0
        N = cluster.N
        for i in range(N):
            rij = cluster.r[i+1:] - cluster.r[i]
            dist = np.linalg.norm(rij, axis=1)
            PE += -G * cluster.m[i] * np.sum(cluster.m[i+1:] / (dist + 1e-10))
        
        return KE, PE, KE + PE

    def detect_collision_pair(self): 

        N = self.cluster.N
        r = self.cluster.r 

        for i in range(N): 
            rij = r[i+1:] - r[i]
            d2 = np.sum(rij**2, axis = 1)
            small = np.where(d2<self.r_c * self.r_c)[0]

            if small.size>0:
                j = i+1 + int(small[0])
                return i, j 
        return None
    def run(self, verbose = None):
        t = 0.0 
        cluster = self.cluster

        results = []

        
        KE0, PE0, E0 = self.compute_energy()
        self.energy_history.append((t, KE0, PE0, E0))

        y = cluster.get_state_vector()

        while t < self.t_end and cluster.N >=2: 
            t_next = min(t + self.dt_event, self.t_end)

            masses =cluster.m.copy() 
            sol = solve_ivp(fun = lambda tt,yy: ode_func(tt, yy, masses), t_span = (t, t_next), y0=y, method = 'RK45', max_step = self.max_step,
                            rtol = 1e-6, atol = 1e-9) 
            
            y = sol.y[:, -1]

            cluster.update_state_vector(y)

            t = sol.t[-1]

            
            KE, PE, E_tot = self.compute_energy()
            self.energy_history.append((t, KE, PE, E_tot))

            if verbose: print(f"[t = {t:4f}] N = {cluster.N}")

            pair = self.detect_collision_pair()

            if pair is not None: 
                i, j = pair 
                p_merge = self.p_merge_func(cluster, i ,j)
                u = self.rng.random()
                if verbose: 
                    vrel = np.linalg.norm(cluster.v[i]- cluster.v[j])
                    print(f" collision detected between {i} and {j}: r = {np.linalg.norm(cluster.r[i]-cluster.r[j]): .4e}, v_rel = {vrel:.4f}, p merge = {p_merge: 3f}, u ={u:.3f}")
                
                if u< p_merge: 

                    if j<i :
                        i,j = j, i
                    
                    cluster.merge_pair(i,j)
                    if verbose:
                        print(f"  merged -> new N= {cluster.N}")
                
                else: 
                    cluster.scatter_pair_elastic(i, j)
                    if verbose: print(f"   scattered(elastic)")

                y = cluster.get_state_vector()
                KE, PE, E_tot = self.compute_energy()
                self.energy_history.append((t, KE, PE, E_tot))

            results.append((t, cluster.m.copy(), cluster.r.copy(), cluster.v.copy()))

        return results
    
class Analysis:
    """Simple analysis utilities."""

    @staticmethod
    def check_momentum_conservation(before_cluster: StarClusters, after_cluster: StarClusters, tol=1e-6):
        """Compute total momentum difference (Note: merges alter N)"""
        P_before = np.sum(before_cluster.m[:, None] * before_cluster.v, axis=0)
        P_after = np.sum(after_cluster.m[:, None] * after_cluster.v, axis=0)
        abs_error = np.linalg.norm(P_before - P_after)
        rel_error = abs_error/(np.linalg.norm(P_before) +1e-12)
        return {
            'P_initial': P_before,
            'P_final': P_after,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'passed': rel_error < tol
        }

    @staticmethod
    def check_mass_conservation(before_cluster: StarClusters, after_cluster: StarClusters):
        """
        Check mass conservation between two cluster states.
        
        """
        M_before = np.sum(before_cluster.m)
        M_after = np.sum(after_cluster.m)
        
        abs_error = abs(M_after - M_before)
        rel_error = abs_error / M_before
        
        return {
            'M_initial': M_before,
            'M_final': M_after,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'passed': rel_error < 1e-10
        }

    @staticmethod
    def analyze_energy_conservation(sim: Simulator):
        if not sim.energy_history:
            return {'error': 'No energy history available'}
        
        times = np.array([e[0] for e in sim.energy_history])
        KE = np.array([e[1] for e in sim.energy_history])
        PE = np.array([e[2] for e in sim.energy_history])
        E_tot = np.array([e[3] for e in sim.energy_history])
        
        E_initial = E_tot[0]
        E_final = E_tot[-1]
        E_drift = E_final - E_initial
        E_drift_pct = 100 * abs(E_drift) / abs(E_initial)
        
        # Find max absolute drift
        E_drift_abs_max = np.max(np.abs(E_tot - E_initial))
        E_drift_pct_max = 100 * E_drift_abs_max / abs(E_initial)
        
        return {
            'times': times,
            'KE': KE,
            'PE': PE,
            'E_tot': E_tot,
            'E_initial': E_initial,
            'E_final': E_final,
            'drift_absolute': E_drift,
            'drift_percent': E_drift_pct,
            'drift_percent_max': E_drift_pct_max,
            'passed': E_drift_pct < 10.0  # 10% is reasonable threshold
        }

    @staticmethod
    def final_mass_stats(cluster: StarClusters):
        """
        Compute statistics of final mass distribution.
        
        Parameters
        ----------
        cluster : StarClusters
            Cluster to analyze
        
        Returns
        -------
        dict
            Dictionary of mass statistics
        """
        ms = cluster.m
        return {
            'N': cluster.N, 
            'min': float(ms.min()), 
            'max': float(ms.max()),
            'mean': float(ms.mean()), 
            'median': float(np.median(ms)),
            'std': float(ms.std())
        }

    @staticmethod
    def print_validation_report(momentum_result, mass_result, energy_result):
        """
        Print formatted validation report.
        
        Parameters
        ----------
        momentum_result : dict
            Output from check_momentum_conservation
        mass_result : dict
            Output from check_mass_conservation
        energy_result : dict
            Output from analyze_energy_conservation
        """
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        # Momentum
        print("\n[1] MOMENTUM CONSERVATION")
        print("-" * 70)
        P_i = momentum_result['P_initial']
        P_f = momentum_result['P_final']
        print(f"  Initial momentum: [{P_i[0]:+.6e}, {P_i[1]:+.6e}, {P_i[2]:+.6e}]")
        print(f"  Final momentum:   [{P_f[0]:+.6e}, {P_f[1]:+.6e}, {P_f[2]:+.6e}]")
        print(f"  Absolute error:    {momentum_result['absolute_error']:.6e}")
        print(f"  Relative error:    {momentum_result['relative_error']:.6e}")
        status = "✓ PASS" if momentum_result['passed'] else "✗ FAIL"
        print(f"  Status: {status}")
        
        # Mass
        print("\n[2] MASS CONSERVATION")
        print("-" * 70)
        print(f"  Initial total mass: {mass_result['M_initial']:.10f}")
        print(f"  Final total mass:   {mass_result['M_final']:.10f}")
        print(f"  Absolute error:     {mass_result['absolute_error']:.6e}")
        print(f"  Relative error:     {mass_result['relative_error']:.6e}")
        status = "✓ PASS (exact)" if mass_result['passed'] else "⚠ WARNING"
        print(f"  Status: {status}")
        
        # Energy
        print("\n[3] ENERGY EVOLUTION")
        print("-" * 70)
        if 'error' in energy_result:
            print(f"  {energy_result['error']}")
        else:
            print(f"  Initial energy:     {energy_result['E_initial']:.10f}")
            print(f"  Final energy:       {energy_result['E_final']:.10f}")
            print(f"  Absolute drift:     {energy_result['drift_absolute']:+.6e}")
            print(f"  Percent drift:      {energy_result['drift_percent']:+.4f}%")
            print(f"  Max percent drift:  {energy_result['drift_percent_max']:.4f}%")
            
            status = "✓ ACCEPTABLE" if energy_result['passed'] else "⚠ HIGH DRIFT"
            print(f"  Status: {status}")
            
            if not energy_result['passed']:
                print("\n  Note: Energy drift > 10% may indicate:")
                print("    - Need for smaller integration timestep")
                print("    - Multiple inelastic mergers")
                print("    - Close encounters causing numerical issues")
        
        print("\n" + "="*70)



def plot_validation_results(results, sim, initial_cluster, save_path='validation_results.png'):
    """
    Create validation plots: energy conservation, momentum conservation, mass distribution.
    
    Parameters
    ----------
    results : list
        Simulation results from sim.run()
    sim : Simulator
        Completed simulator instance
    initial_cluster : StarClusters
        Initial cluster state for comparison
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Mass distribution evolution
    ax = axes[0]
    initial_masses = initial_cluster.m
    final_masses = results[-1][1]
    
    bins = np.linspace(
        min(initial_masses.min(), final_masses.min()),
        max(initial_masses.max(), final_masses.max()),
        20
    )
    
    ax.hist(initial_masses, bins=bins, alpha=0.6, label='Initial', edgecolor='black')
    ax.hist(final_masses, bins=bins, alpha=0.6, label='Final', edgecolor='black')
    ax.set_xlabel('Mass', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Mass Distribution Evolution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Energy conservation
    ax = axes[1]
    energy_data = Analysis.analyze_energy_conservation(sim)
    if 'error' not in energy_data:
        t_e = energy_data['times']
        E_tot = energy_data['E_tot']
        E_initial = energy_data['E_initial']
        E_rel = 100 * (E_tot - E_initial) / abs(E_initial)
        
        ax.plot(t_e, E_rel, 'o-', markersize=4, linewidth=2, color='blue')
        ax.axhline(0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Perfect conservation')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy Drift (%)', fontsize=12)
        ax.set_title('Energy Conservation', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 3. Momentum conservation (show components)
    ax = axes[2]
    momentum_check = Analysis.check_momentum_conservation(initial_cluster, sim.cluster)
    
    P_i = momentum_check['P_initial']
    P_f = momentum_check['P_final']
    
    components = ['x', 'y', 'z']
    x_pos = np.arange(len(components))
    width = 0.35
    
    ax.bar(x_pos - width/2, P_i, width, label='Initial', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, P_f, width, label='Final', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Momentum', fontsize=12)
    ax.set_title('Momentum Conservation', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add error text
    rel_err = momentum_check['relative_error']
    ax.text(0.95, 0.95, f"Rel. error: {rel_err:.2e}", 
            transform=ax.transAxes, 
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()

def demo_run():
    """Run a demonstration simulation with validation."""
    N = 50
    
    print("="*70)
    print("N-BODY STAR CLUSTER SIMULATION")
    print("="*70)
    
    print("\n[1/5] Initializing cluster...")
    cluster = StarClusters.initialize_random(
        N=N, 
        mass_alpha=2.35, 
        mass_min=0.5, 
        mass_max=3.0,
        radius_scale=1.0, 
        velocity_scale=0.5, 
        random_state=42
    )
    
    # Store initial state for validation
    initial_cluster = StarClusters(
        cluster.m.copy(), 
        cluster.r.copy(), 
        cluster.v.copy()
    )
    
    print(f"  Initial N: {N}")
    print(f"  Total mass: {cluster.m.sum():.3f}")
    
    print("\n[2/5] Running simulation...")
    sim = Simulator(
        cluster, 
        t_end=0.5, 
        r_c=0.05, 
        dt_event=0.01, 
        max_step=0.005, 
        rng_seed=123
    )
    
    t0 = time.time()
    results = sim.run(verbose=True)
    runtime = time.time() - t0
    
    print(f"\n[3/5] Simulation completed in {runtime:.2f}s")
    
    final_cluster = sim.cluster
    n_mergers = N - final_cluster.N
    
    print("\n[4/5] Final state:")
    print(f"  Final N: {final_cluster.N}")
    print(f"  Mergers: {n_mergers}")
    
    print("\n[5/5] Validation:")
    momentum_check = Analysis.check_momentum_conservation(initial_cluster, final_cluster)
    mass_check = Analysis.check_mass_conservation(initial_cluster, final_cluster)
    energy_check = Analysis.analyze_energy_conservation(sim)
    
    Analysis.print_validation_report(momentum_check, mass_check, energy_check)
    
    print("\nGenerating validation plots...")
    plot_validation_results(results, sim, initial_cluster)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
    return results, sim, initial_cluster


if __name__ == "__main__":
    demo_run()