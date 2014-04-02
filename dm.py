"""Design matrix processing functions"""
import numpy as np
import pandas as pd

from copy import deepcopy
from statsmodels.api import GLS


def convolve_hrf(dm, hrf, cols=None):
    """Convolve hrf onto design matrix columns.
    
    Note
    ----
    If cols is not None, only the specified cols
    are returned.
    """
    
    if cols is None:
        cols = range(dm.shape[1])
    
    # Select data
    try:
        dm_c = dm[:,cols]  ## np
    except TypeError:
        dm_c = dm[cols]    ## pd
    
    # Assume 2d, fall back to 1d.
    try:
        for col in cols:
            try:
                dm_c[:,col] = np.convolve(dm[:,col], hrf)[0:dm.shape[0]] ## np
            except TypeError:
                dm_c[col] = np.convolve(dm[col], hrf)[0:dm.shape[0]]     ## pd
    except IndexError:
        dm_c = np.convolve(dm[:], hrf)[0:dm.shape[0]]
            ## cols don't matter for 1d
    
    return dm_c


def orthogonalize(dm, cols):
    """ Orthogonalize dm cols (by regression). 
    
    Parameters
    ---------
    dm : array-like (n_samples, n_cond)
        The design matrix
    cols : list
        A list of cols to orthogonalize pair-wise
        moving rightward
    """
    
    if dm.ndim == 1:
        return dm
    if len(cols) < 2:
        raise ValueError("cols must have two elements")
    
    orth_dm = dm.copy()
    for j in range(len(cols)-1):
        left = cols[j]
        right = cols[j+1]
        
        # Orthgonalize left to right....
        ## GLS(y
        try: 
            glm = GLS(dm[:,right], dm[:,left]).fit()    ## np
            orth_dm[:,right] = glm.resid
        except TypeError:
            glm = GLS(dm[right], dm[left]).fit()        ## pd
            orth_dm[right] = glm.resid
        
    return orth_dm


def add_movement(dm, movement):
    """Add a movement matrix to the design matrix
    
    Parameters
    ---------
    dm : array-like (n_samples, n_cond)
        The design matrix
    movement : array-like (n_samples, n_movement_regressors)
        The movement matrix
    """
    
    if dm.shape[0] != movement.shape[0]:
        raise ValueError("Wrong n_samples in dm or movement")
    
    try:
        return pd.concat([dm, movement], axis=1)
    except AttributeError:
        return np.hstack((dm, dm_movement))

