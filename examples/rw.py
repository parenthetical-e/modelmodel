import os
import shutil
import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from patsy.builtins import scale

from modelmodel.behave import behave
from modelmodel.behave import reinforce
from modelmodel.noise import white
from modelmodel.hrf import double_gamma as dg
from modelmodel.dm import convolve_hrf
from modelmodel.dm import orthogonalize
from modelmodel.io import reformat_model
from modelmodel.io import read_models
from modelmodel.io import reformat_contrast
from modelmodel.io import merge_results
from modelmodel.io import write_hdf
from modelmodel.bold import create_bold


parser = argparse.ArgumentParser(
        description="Simulate a Rescorla Wagner experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        "name", type=str,
        help="Name of this exp"
        )
parser.add_argument(
        "-N", 
        default=1000, type=int,
        help="Number of iterations"
        )         
parser.add_argument(
        "--n_trials",
        default=60, type=int,
        help="Number of trials / cond"
        )
parser.add_argument(
        "--behave",
        default='learn',
        help="Behavior mode ('learn', 'random')"
        )
parser.add_argument(
        "--models", type=str, nargs=1,
        help="The models file (.ini)"
        )
parser.add_argument(
        "--save_behave", type=bool, default=False,
        help="Save the behave data?"
        )
parser.add_argument(
        "--seed",
        default=42, type=int,
        help="RandomState seed"
        )
args = parser.parse_args()
prng = np.random.RandomState(args.seed)
n_cond = 1 

# Get and parse the model.ini file
model_configs = read_models(args.models)

# Pick which data to use as BOLD
asbold = ['box', 'acc', 'p', 'value', 'rpe', 'rand']

# Regress for each BOLD, and model for N interations 
results = {}
for n in range(args.N):
    print("Iteration {0}".format(n))
    
    # Create data
    if args.behave == 'learn':
        trials, acc, p, prng = behave.learn(
                n_cond, args.n_trials, 
                loc=prng.normal(3, .3), prng=prng
                )
    elif args.behave == 'random':
        trials, acc, p, prng = behave.random(
                n_cond, args.n_trials, prng=prng
                )
    else:
        raise ValueError('--behave not understood')
    df, rlpars = reinforce.rescorla_wagner(trials, acc, p, prng=prng)

    # Convolve with HRF
    df = convolve_hrf(df, dg(), asbold)
    
    # Orth select regressors
    to_orth = [['box', bold] for bold in asbold if bold != 'box']
    for orth in to_orth:
        df[orth[1]+'_o'] = orthogonalize(df, orth)[orth[1]]
    
    # Do the regressions    
    n_results = {}
    for model_name, model, test, hypoth in zip(*model_configs):
        for bold_name in asbold:
            l = df.shape[0]
            noi, prng = white(l, prng=prng)

            df['bold'] = create_bold([df[bold_name].values], None, noi)

            smo = smf.ols(model, data=df).fit()
            print(smo.summary2())
            
            stato = None
            if test == 't':
                stato = smo.t_test(hypoth)
            elif test == 'F':
                stato = smo.f_test(hypoth)
            elif test is not None:
                raise ValueError("Unknown test")
            
            savedf = None
            if args.save_behave: 
                savedf = df 

            n_results.update(merge_results(
                    'bold:'+bold_name + '_' + 'model:'+model_name,
                    model, smo, df=savedf, stato=stato, other=rlpars
                    ))
                    
    results.update({str(n) : n_results})

write_hdf(results, str(args.name))
