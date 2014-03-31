import numpy as np
import scipy.stats as stats
from modelmodel.misc import process_prng


def random(N, prng=None):
    """Create N p(correct) sampled from a uniform distribution (0-1)"""
    
    prng = process_prng(prng)
    return prng.rand(N), prng


def learn(N, loc=3, prng=None):
    """Simulate learning moving p(correct) from random to a sigmoid
    
    Note
    ----
    * Transition from random to learn is itself random, sampled from 
        uniform(1, N).
    * loc of 3 gives 'realistic' learning curves
    """

    prng = process_prng(prng)
    np.random.set_state(prng.get_state())
    
    # Random and learn are divided by T
    T = int(prng.randint(0, N, 1))
    
    # Random p
    p_1, prng = random(T, prng)
    
    # Learn p
    trials = np.arange(.01, 10, (10/float(N - T)))
    trials = trials + stats.norm.rvs(size=trials.shape[0]) 
    p_2 = stats.norm.cdf(trials,loc)
    p_2[p_2 < 0.5] = prng.rand(np.sum(p_2 < 0.5))
        ## Remove p vals les than 0.5 otherwise
        ## p drops badly as learning is suppose
        ## to start
        
    prng.set_state(np.random.get_state())

    return np.concatenate([p_1, p_2]), prng

