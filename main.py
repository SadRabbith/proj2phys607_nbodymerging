import numpy as np 
import matplotlib.pyplot as plt
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
        u = range_val(size)

        if alpha == 1.0: 
            np.exp(u * (np.log(xmax) - np.log(xmin)) + np.log(xmin))
        
        powl = 1.0 - alpha
        min = xmin ** powl 
        max = xmax ** powl 

        x = (u * (max - min) +min)**(1 / powl) 

        return x 
    
    @staticmethod
    def rejection_sampling_MB(scale, size = 1, vmax = None, random_state = None):
        range_val = np.random.default_rng(random_state)
        if vmax is None: 
            vmax = scale * 6
        sample = [] 

        n = size 
        vx = range_val.normal(scale = scale, size =n)
        vy = range_val.normal(scale = scale, size = n)
        vz = range_val.normal(scale = scale, size =n) 
        speeds = np.sqrt(vx**2 + vy**2 + vz**2) 

        return speeds
    
class StarClusters: 
    """stores star properties""" 

    def _init_(self, masses, positions, velocities):

        assert masses.shape[0] == positions.shape[0] == velocities.shape[0] 

        self.m = masses.astype(float) 
        self.r = positions.astype(float) 
        self.v = velocities.astype(float) 
        self.N = self.m.size 
    
    @staticmethod
    def initialize_random(cls, N, mass_alpha = 2.35, mass_min = 0.5, mass_max = 5.0, radius_scale = 1.0, 
                          velocity_scale = 0.5, random_scale = None): #MC methods to find masses radial positions and velocities
        
        range_val = np.random.default_rng(random_scale)
        masses = SamplingTool.inverse_cdf_powerlaw(alpha = mass_alpha, xmin = mass_min, xmax = mass_max, size = N, random_state=range_val)

        u = range_val.random(size = N) 
        pos = range_val.normal(scale=radius_scale, size = (N,3)) 

        pos = pos / np.linalg.norm(pos, axis = 1)[;, None] *(np.abs(range_val.normal(radius_scale, size =N))[:, None]) 

        pos = range_val.normal(scale = radius_scale, size= (N,3)) 

        speeds = SamplingTool.rejection_sampling_MB(scale = velocity_scale, size = N, random_state= range_val) 

        theta = np.arccos(1 - 2 *range_val.random(size = N)) 
        phi = 2 * np.pi * range_val. random(size = N) 
        vx = speeds * np.sin(theta) * np.cos(phi) 
        vy = speeds* np.sin(theta) * np.sin(phi) 
        vz = speeds * np.cos(theta) 

        vel = np.vstack(masses = masses, positions = pos, velocities = vel)


    def get_state_vector(self):
        """Return flattened state vector y = [r.flatten(), v.flatten()]"""
        y = np.concatenate([self.r.ravel(), self.v.ravel()])
        return y


    def update_state_vector(self,y):

        N = self.N 

        r = y[:3*N].reshapre((N,3))
        v = y[3*N].reshapre((N,3)) 
        self.r, self.v = r.copy(), v.copy() 

    def remove_index(self, idx): 

        mask = np.ones(self.N, dtype = bool) 
        mask[idx] = False
        self.m, self.r, self.v, self.N = self.m[mask], self.r[mask], self.v[mask], self.N = self.m.size

    



    



