"""Create a table of simulated RL (Rescorla Wagner) data"""
import os
import argparse
import pandas as pd
import numpy as np
from modelmodel.behave import behave
from modelmodel.behave import reinforce
from modelmodel.hrf import double_gamma as dg
from modelmodel.dm import convolve_hrf


parser = argparse.ArgumentParser(
        description="Create a table of simulated RL (Rescorla Wagner) data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        "name", type=str,
        help="Name of table"
        )
parser.add_argument(
        "N", type=int,
        help="Number of samples"
        )
parser.add_argument(
        "--behave", type=str, default='learn',
        help="Behavior learning mode"
        )
parser.add_argument(
        "--n_cond", type=int, default=1,
        help="N cond"
        )
parser.add_argument(
        "--n_trials", type=int, default=60,
        help="N trials/cond"
        )
parser.add_argument(
        "--alpha",
        type=float, default=None,
        help="Set alpha values"
        )
parser.add_argument(
        "--convolve", type=bool, default=False,
        help="Convolve each col with the (canonical) double gamma HRF"
        )
parser.add_argument(
        "--seed",
        default=42, type=int,
        help="RandomState seed"
        )
args = parser.parse_args()
prng = np.random.RandomState(args.seed)

dfs = []
for n in range(args.N):
    if args.behave == 'learn':
        trial, acc, p, prng = behave.learn(
                args.n_cond, args.n_trials, 
                loc=prng.normal(3, .3), prng=prng
                )
    elif args.behave == 'random':
        trial, acc, p, prng = behave.random(
                args.n_cond, args.n_trials, prng=prng
                )
    else:
        raise ValueError('--behave not understood')
    
    df, rlpars = reinforce.rescorla_wagner(
            trial, acc, p, alpha=args.alpha, prng=prng
            )
    del df['rand']
    
    l = trial.shape[0]
    df['count'] = np.repeat(n, l)
    df['index'] = np.arange(l, dtype=np.int)
    
    dfs.append(df)
df = pd.concat(dfs, axis=0)

if args.convolve:
    tocon = ['box', 'acc', 'p', 'rpe', 'value']
    condf = convolve_hrf(df, dg(), tocon)
    for con in tocon:
        df[con] = condf[con]
    
df.to_csv(args.name, index=False, float_format='%.8f')
