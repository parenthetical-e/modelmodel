"""Monte Carlo simulations of accuracy"""
import numpy as np
import scipy.stats as stats
from modelmodel.misc import process_prng


def accuracy(p, prng=None):
    prng = process_prng(prng)
    
    N = len(p)
    return prng.binomial(1, p, N), prng


