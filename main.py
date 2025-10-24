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
        
