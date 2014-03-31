"""A collection of hemodynamic response functions."""
import numpy as np
import scipy.stats as stats
from modelmodel.misc import process_prng

    
def double_gamma(width=32, TR=1, a1=6.0, a2=12., b1=0.9, b2=0.9, c=0.35):
    """
    Returns a HRF.  Defaults are the canonical parameters.
    """
    
    x_range = np.arange(0, width, TR)    
    d1 = a1 * b1
    d2 = a2 * b2
    
    hrf = ((x_range / d1) ** a1 * np.exp((d1 - x_range) / b1)) - (
            c * (x_range / d2) ** a2 *np.exp((d2 - x_range) / b2))
    
    return hrf


def _preturb(weight, width=32, TR=1, a1=6.0, a2=12., b1=0.9, b2=0.9, c=0.35, prng=None):    
    prng = process_prng(prng)    
    np.random.set_state(prng.get_state())

    # Parameters to preturb
    params = {a1:6.0, a2:12.0, b1:0.9, b2:0.9, c:0.35}

    # Preturb it
    keys = params.keys()
    prng.shuffle(keys)
    par = params[keys[0]]
    params[keys[0]] = prng.normal(loc=par, scale=par / (1. * weight))
    
    # Add unpreturbed params
    params['width'] = width
    params['TR'] = TR
    
    return params, prng


def preturb_double_gamma(weight, width=32, TR=1, a1=6.0, a2=12., b1=0.9, b2=0.9, c=0.35, prng=None):
    """
    Returns a (normally) perturbed HRF.  
    
    Defaults are the canonical parameters. Degree of perturbation 
    can be rescaled by weight.
    """
    
    params, prng = _preturb(
            weight, width=32, TR=1, a1=6.0, a2=12., b1=0.9, 
            b2=0.9, c=0.35, prng=None
            )
    
    return double_gamma(**params), prng


def fir(events, bold, window_size=30):
    raise NotImplementedError("Create me")

