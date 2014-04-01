"""Create a table of simulated behave data"""
import os
import argparse
import pandas as pd
import numpy as np
from modelmodel.behave import behave


parser = argparse.ArgumentParser(
        description="Create a table of simulated behave data",
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
        "--seed",
        default=42, type=int,
        help="RandomState seed"
        )
args = parser.parse_args()
prng = np.random.RandomState(args.seed)

trials = []
ps = []
accs = []
count = []
index = []
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
    
    trials.append(trial)
    ps.append(p)    
    accs.append(acc)
    
    l = trial.shape[0]
    count.append(np.repeat(n, l))
    index.append(np.arange(l, dtype=np.int))

count = np.concatenate(count)
index = np.concatenate(index)
trials = np.concatenate(trials)
accs = np.concatenate(accs)
ps = np.concatenate(ps)
data = np.vstack([count, index, trials, accs, ps]).transpose()

df = pd.DataFrame(data=data, columns=('count', 'index', 'trials', 'acc', 'p'))
df.to_csv(args.name, index=False, float_format='%.8f')
