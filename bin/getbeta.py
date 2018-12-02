"""Run conjunction tests on csv tables of stats and p-values."""
import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(
        description="Extract the betas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
        "name", type=str,
        help="Name of file"
)
parser.add_argument(
        "stat_values", type=str,
        help="Name of statistics file"
)
parser.add_argument(
        "p_values", type=str, default=None,
        help="Name of p-values file"
)
parser.add_argument(
        "--col", type=int, default=-1,
        help="Column to extract"
)
parser.add_argument(
        "--missing_value", type=int, default=-999999,
        help="Missing value code",
)
parser.add_argument(
        "--drop", type=int, default=None,
        help="Drop this column number from the analysis",
)
args = parser.parse_args()

stats = pd.read_csv(args.stat_values)
ps = pd.read_csv(args.p_values)

if args.drop is not None:
    stats = stats.drop([stats.columns[args.drop]], 1)
    ps = ps.drop([ps.columns[args.drop]], 1)

selected = []
p_selected = []
paths = []
for i in range(stats.shape[0]):
    spath = stats.ix[i,-1]  # get only path
    sdata = stats.ix[i,0:-1].values  # get all but path
    sdata = sdata[sdata != args.missing_value]

    ppath = ps.ix[i,-1]
    pdata = ps.ix[i,0:-1].values
    pdata = pdata[pdata != args.missing_value]

    if sdata.shape[0] != pdata.shape[0]:
        raise ValueError("Row lengths don't match ({0})".format(i))

    stat = sdata[args.col]
    p = pdata[args.col]

    selected.append(stat)
    p_selected.append(p)
    paths.append(spath)

df = pd.DataFrame(
        data=np.vstack([selected, p_selected]).transpose(),
        columns=['value', 'p']
        )
df['path'] = paths
df.to_csv(args.name, index=False, float_format='%.8f')

