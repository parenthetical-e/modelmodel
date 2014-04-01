"""Extract data from rw.py experiments to a .csv file"""
import os
import argparse
import h5py
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
        description="Simulate a Rescorla Wagner experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        "--hdf", type=str,
        help="Name of hdf5 file"
        )
parser.add_argument(
        "--names", type=str, nargs='+',
        help="Names of .csv"
        )
parser.add_argument(
        "--paths", type=str, nargs='+',
        help="Paths of data to traverse. Use '*' to select all leaves"
        )
parser.add_argument(
        "--dims", type=int, nargs='+',
        help="Dimensionality of data expected for each path"
        )
args = parser.parse_args()


# Arg processing
hdf = h5py.File(args.hdf, 'r')
names = args.names
paths = args.paths
dims = args.dims

for dim in dims:
    if dim <= 0: raise ValueError("dim can't be 0 or negative")
    if dim > 2: raise ValueError("dim can't be greater than 2")
    
# Build up the tree to traverse
# get the data and write it
for path, name, dim in zip(paths, names, dims):
    parts = path.split('*')
    tree = None
    for part in parts[:-1]:  ## Hide last to avoid calling .keys() on it
        if tree is None:
            tree = hdf[part].keys()
        else:
            tree = [
                    os.path.join(leave, key) 
                    for leave in tree 
                    for key in hdf[leave].keys()
                    ]
    tree = [leave + parts[-1] for leave in tree] ## Add the stat to extract

    extracted = []
    for leave in tree:
        extracted.append(hdf[leave].value)
    
    undertree = []
    for leave in tree:
        parts = leave.split('/')
        undertree.append('_'.join(parts))
    
    if dim == 1:
        data = np.squeeze(np.asarray(extracted))
        df = pd.DataFrame(data={parts[-1] : data})
        df['path'] = undertree
    else:
        data = np.vstack(extracted).transpose()
        ncol = data.shape[1]
        df = pd.DataFrame(data=data, cols=undertree)
    df.to_csv(name, index=False, float_format='%.6f')
