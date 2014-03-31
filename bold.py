import numpy as np
from modelmodel.dm import convolve_hrf
from modelmodel.misc import process_prng


def create_bold(arrays, hrf=None, noise=None):
    """ 
    Sum the array(s) of data and (optionally) convolve with an hrf, 
    add (optional) noise, creating a (mock) bold signal.
    """
    
    arr = np.vstack(arrays).transpose()
    if arr.shape[1] == 1:
        bold = arr
    else:
        bold = arr.sum(axis=1)
    bold = bold.flatten()

    if hrf is not None:
        bold = convolve_hrf(bold, hrf)
    
    if noise is not None:
        bold = bold + noise
    
    return bold
