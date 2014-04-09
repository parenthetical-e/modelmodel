"""Create lists of randomized trials."""
import os
import numpy as np

from modelmodel.misc import process_prng


def jitter(trials, code=0, fraction=.5, jit=(1,7), prng=None):
    """Introduce random (uniform) jitter to trials"""
    
    prng = process_prng(prng)
    
    jittimes = np.arange(*jit, dtype=np.int)
    jittered = []
    for t in trials:
        if fraction >= prng.rand(1):
            prng.shuffle(jittimes)
            jt = [t, ] + [code, ] * jittimes[0]
            jittered.extend(jt)
        else:
            jittered.append(t)
    
    return np.asarray(jittered), prng
        

def random(N, k, prng=None):
    """Creates a randomly list of trials (int) of N cond with 
    k trials / cond.
    """

    prng = process_prng(prng)
    
    trials = []
    [trials.extend([n, ] * k) for n in range(1, N+1)]
    
    trials = np.asarray(trials)
    prng.shuffle(trials) 

    return trials, prng


