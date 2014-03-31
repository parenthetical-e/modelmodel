""" Noise models. """
import numpy as np
import scipy.stats as stats
from simfMRI.misc import process_prng


def white(N, sigma=1, prng=None):
    """ Create white noise.
    
    Parameters
    ---------
    N : numeric
        Length of 1d noise array to return
    sigma : numeric
        Standard deviation
    prng : np.random.RandomState, None
        A RandomState instance, or None
    """
    
    prng = process_prng(prng)
    
    return prng.normal(loc=0, scale=sigma, size=N), prng


def ar1(N, alpha=0.2, sigma=1, prng=None):
    """ Create AR1 noise.
    
    Parameters
    ---------
    N : numeric
        Length of 1d noise array to return
    alpha : float
        Degree of autocorrelation
    sigma : numeric
        Standard deviation of white noise
    prng : np.random.RandomState, None
        A RandomState instance, or None
        
    Alpha of 0.2 was taken from the 'temporalnoise.R' function 
    in the R neuRosim package (ver 02-10)
    """
    
    if (alpha > 1) or (alpha < 0):
        raise ValueError("alpha must be between 0-1.")
    
    prng = process_prng(prng)
    
    noise, prng = white(N=N, sigma=sigma, prng=prng)
    arnoise = [noise[0], ]
    
    [arnoise.append(
            noise[ii] + (alpha * noise[ii-1])
            ) for ii in range(1, len(noise))]
    
    return arnoise, prng


def physio(N, TR=1, freq_heart=1.17, freq_resp=0.2, sigma=1, prng=None):
    """ Create periodic physiological noise 
    
    Parameters
    ---------
    N : numeric
        Length of 1d noise array to return
    TR : float
        The repetition rate (BOLD signal)
    freq_heart : float
        Frequency of heart rate (s)
    freq_resp : float
        Frequency of respiration (s)
    sigma : numeric
        Standard deviation of white noise
    prng : np.random.RandomState, None
        A RandomState instance, or None
    
    Note
    ----  
    freq_heart and freq_resp defaults were taken from 
    the 'temporalnoise.R' function in the R neuRosim 
    package (ver 02-10)
    """
        
    # Calculate rates
    heart_beat = 2 * np.pi * freq_heart * TR
    resp_rate = 2 * np.pi * freq_resp * TR
    
    # Use rate to make periodic heart 
    # and respiration (physio) drift 
    # timeseries
    t = np.arange(N)
    hr_drift = np.sin(heart_beat * t) + np.cos(resp_rate * t)
    
    # Renormalize sigma using the 
    # sigma of the physio signals
    hr_weight = sigma / np.std(hr_drift)
    
    # Create the white noise then
    # add the weighted physio
    noise, prng = white(N=N, prng=prng) 
    noise += hr_weight * hr_drift
    
    return noise, prng


def _gen_drifts(nrow, ncol):
    idx = np.arange(0, nrow)

    drifts = np.zeros((nrow, ncol+1))
    drifts[:,0] = np.repeat(1 / np.sqrt(nrow), nrow)
    for col in range(2, ncol+1):
        drift = np.sqrt(2. / nrow) * 10. * np.cos(
                np.pi * (2. * idx + 1.) * (col - 1.) / (2. * nrow))
        drifts[:, col] = drift
        
    return drifts
    
    
def lowfreqdrift(N, TR=1, sigma=1, prng=None):
    """ Create noise with a low frequency drift (0.002-0.015 Hz)  
    
    Parameters
    ---------
    N : numeric
        Length of 1d noise array to return
    TR : float
        The repetition rate (BOLD signal)
    prng : np.random.RandomState, None
        A RandomState instance, or None
    
    Note
    ----  
    Smith et al (1999), Investigation of the Low Frequency Drift in fMRI 
    Signal, NeuroImage 9, 526-533.
    
    This function was ported form a similar function ('lowfreqdrift.R')
    in the R 'neuRosim' package (ver 02-10):
    
    http://cran.r-project.org/web/packages/neuRosim/index.html
    """

    prng = process_prng(prng)
    
    freq = prng.randint(66, 500)
        ## i.e. 0.002-0.015 Hz
    
    nbasis = int(np.floor(2 * (N * TR) / freq + 1))
    noise = _gen_drifts(N, nbasis)
    noise = noise[:,1:]     ## Drop the first col
    noise = noise.sum(1)    ## Sum the rows, creating
                            ## creating the final noise
    
    # Now add white noise
    whiten, prng = white(N, sigma=sigma, prng=prng)
    noise += whiten

    return noise, prng


def onef(N, fraction, prng=None):
    """ Simulate the typical 1/f fMRI noise distribution.
    """
    
    raise NotImplementedError("TODO")

