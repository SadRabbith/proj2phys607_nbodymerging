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
        elastic scattering in the two-body approximation:
        - Transform to center-of-mass frame, reflect relative velocity across line-of-centers randomly
        - This is an approximation; in a real two-body scattering you'd solve for post-scatter velocities
          given impact parameter and scattering law.
        """
        m1, m2 = self.m[i], self.m[j]
        r_rel = self.r[i] - self.r[j]
        rhat = r_rel / (np.linalg.norm(r_rel) + 1e-12)
        v1, v2 = self.v[i].copy(), self.v[j].copy()
        v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)
        u1 = v1 - v_cm
        u2 = v2 - v_cm
        # reflect the component along rhat for each particle (simple model)
        u1_par = np.dot(u1, rhat) * rhat
        u1_perp = u1 - u1_par
        u2_par = np.dot(u2, rhat) * rhat
        u2_perp = u2 - u2_par
        # swap parallel components (approx elastic exchange)
        u1_par, u2_par = u2_par, u1_par
        self.v[i] = v_cm + u1_par + u1_perp
        self.v[j] = v_cm + u2_par + u2_perp

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
        diff = np.linalg.norm(P_before - P_after)
        return diff

    @staticmethod
    def final_mass_stats(cluster: StarClusters):
        ms = cluster.m
        return {'N': cluster.N, 'min': float(ms.min()), 'max': float(ms.max()),
                'mean': float(ms.mean()), 'median': float(np.median(ms))}


def demo_run():
    """Run a short demo with N ~ 50 to show the pipeline."""
    N = 100
    print("Initializing cluster...")
    cluster = StarClusters.initialize_random(N=N, mass_alpha=2.35, mass_min=0.5, mass_max=3.0,
                                            radius_scale=1.0, velocity_scale=0.1, random_state=42)
    sim = Simulator(cluster, t_end=0.5, r_c=0.05, dt_event=0.01, max_step=0.005, rng_seed=123)
    t0 = time.time()
    results = sim.run(verbose=True)
    t1 = time.time()
    print(f"Simulation finished in {t1 - t0:.2f}s, steps recorded: {len(results)}")
    final_cluster = sim.cluster
    print("Final mass stats:", Analysis.final_mass_stats(final_cluster))
    # Optionally plot final mass histogram if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        m = final_cluster.m
        plt.figure()
        plt.hist(m, bins='auto')
        plt.title("Final Mass Distribution")
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    demo_run()     