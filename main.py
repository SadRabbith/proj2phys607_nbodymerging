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