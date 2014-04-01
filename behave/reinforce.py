"""Simulate RL datasets"""
import numpy as np
import pandas as pd

import rl
from modelmodel.misc import process_prng


def rescorla_wagner(trials, acc, p, alpha=None, prng=None):
    """Create a RW learning dataset
    
    Parameters
    ----------
    trials : list([int, ])
        Trials coded by condition
    acc : list([int((0,1)), ])
         Accuracy data
    p : list([float])
        p(correct)
    alpha : float, None
        The learning rate (None fits by ML)
    """
    
    prng = process_prng(prng)
    l = trials.shape[0]
    
    # fit RW models?
    if alpha is None:
        best_rl_pars, best_logL = rl.fit.ml_delta(acc, trials, 0.05)
        v_dict, rpe_dict = rl.reinforce.b_delta(acc, trials, best_rl_pars[0])
    else:
        best_rl_pars, best_logL = None, None
        v_dict, rpe_dict = rl.reinforce.b_delta(acc, trials, alpha)
        
    values = rl.misc.unpack(v_dict, trials) ## Reformat from dict to array
    rpes = rl.misc.unpack(rpe_dict, trials)
    
    # Store sim data
    box = np.zeros_like(trials)
    box[trials > 0] = 1
    rand = prng.rand(l)
    
    df = pd.DataFrame(data={
                'trials' : trials,
                'box' : box,
                'acc' : acc,
                'p' : p,
                'value' : np.asarray(values),
                'rpe' : np.asarray(rpes),
                'rand' : rand
                })
        
    return df, {'best_rl_pars' : best_rl_pars, 'best_logL' : best_logL}
    