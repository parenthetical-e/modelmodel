""" Functions for reading and writing of model files and results """
from copy import deepcopy
import csv
import h5py
import ConfigParser
import numpy as np
import nibabel as nb


def read_models(ini):
    """Read and parse a model.ini file"""
    conf = ConfigParser.SafeConfigParser()
    readname = conf.read(ini)
        ## If conf.read() can't find model_config
        ## it (annoyingly) returns an empty list

    model_names = []
    models = []
    tests = []
    hypotheses = []
    if readname:
        for sec in conf.sections():
            model_names.append(sec)
            models.append(conf.get(sec, "model"))
            
            # Optional - if a test is chosen a
            # hypothesis must follow
            if conf.has_option(sec, "test"):
                tests.append(conf.get(sec, "test"))
                hypotheses.append(conf.get(sec, "hypothesis"))
            else:
                tests.append(None)
                hypotheses.append(None)            

    else:
        raise IOError("{0} not found".format(model_config))
    
    return model_names, models, tests, hypotheses


def _walkd(dictionary, hdf):
    """Recursively walk the dictionary, adding to you go hdf"""
    for key, val in dictionary.items():
        if isinstance(val, dict):
            hdfnext = hdf.create_group(key)
            _walkd(val, hdfnext)
                ## gettin' all recursive and shit, yo.
        else:
            if val is None: 
                val = 0
                ## h5py does not know 
                ## what to do with None.
            
            data = np.array(val)
            hdf.create_dataset(key, data=data)


def write_hdf(results, name):
    """ Save a list of results (dict) to an HDF5 file.
    
    Parameters
    ----------
    results : dict 
        A results 
    name : str
        The name of the HRF5 file
    """

    hdf = h5py.File(name, 'w')
    for key, val in results.items():
        hdf_ii = hdf.create_group(key)
        _walkd(val, hdf_ii)
    
    hdf.close()


def write_nifti(nifti, name):
    """Write nifiti1 data."""
    
    nb.save(nifti, name)     


def get_hdf_data(name, path):
    """Get data from a HDF5 file assuming there is replicates
    of data branching from the top-level nodes
    """

    # First locate the dataset in the first result,
    # then grab that data for every run.
    hdf = h5py.File(name, 'r')
    
    return [hdf[node + path].value for node in sorted(hdf.keys())]


def get_hdf_data_inc(name, path):
    """ Incrementally get data from a HDF5 file assuming there is replicates
    of data branching from the top-level nodes
    """

    # First locate the dataset in the first result,
    # then grab that data for every run.
    hdf = h5py.File(name, 'r')

    for node in sorted(hdf.keys()):
        yield hdf[node + path].value


def read_nifti(name):
    """Read nifti1 data"""

    return nb.nifti1.load(name)


def read_trials(name):
    """Read 1 col .csv file of trial codes"""
    
    # Open a file handle to <name>
    # slurp it up with csv, and put
    # it in a list.  Returning that
    # list.
    fhandle = open(name, 'r')
    trials_f = csv.reader(fhandle).next()
    trials = [int(trl) for trl in trials_f]
    fhandle.close()

    return trials


def get_model_names(hdf5_name):
    """ Return the model names.
    
    Note
    ----
    Assumes data was written with write_hdf(_inc) 
    """
    
    hdf = h5py.File(hdf5_name, 'r')
    topkeys = hdf.keys()
    models = sorted(hdf[topkeys[0]].keys())       
            ## Need only to check the first node
    
    return models


def get_metadata(hdf5_name, model_name):
    """ Get the BOLD and DM info.
    
    Note
    ----
    Assumes data was written with write_hdf(_inc)
    """

    # Open,
    hdf = h5py.File(hdf5_name, 'r')
    
    # and get its metadata.
    meta = {}
    meta['bold'] = hdf['/1/' + model_name + '/data/meta/bold'].value
    meta['dm'] = hdf['/1/' + model_name + '/data/meta/dm'].value

    return meta


def get_roi_names(hdf5_name):
    """ Get a list of the rois.
    
    Note
    ----
    Assumes data was written with write_hdf(_inc)
    """
    
    return get_hdf_data(hdf5_name, '/batch_code')


def write_all_scores_as_df(hdf5_name, code):
    """Write regression model scores to a .csv table.
    
    Parameters
    ----------
    hdf5_name : str
        Name of the HRF5 file
    code : str, int, float
        ???
        """
    from copy import deepcopy
    
    model_names = get_model_names(hdf5_name)
    roi_names = get_roi_names(hdf5_name)
    score_names = ['bic', 'aic', 'llf', 'r', 'r_adj', 'fvalue', 
            'f_pvalue', 'df_model','df_resid']

    dataframe = []
        ## Score names will be the header
        ## of the dataframe

    for model in model_names:
        data = np.zeros((len(roi_names), len(score_names)))
            ## Will hold only the numerical scores
        
        for col, score in enumerate(score_names):
            path = '/'.join(['', model, score])
            data[:,col] = get_hdf_data(hdf5_name, path)
        
        for row, roi in zip(data, roi_names):
            # Build ech row of the dataframe...
            # Combine data with the metadata 
            # for that model/roi/etc
            meta = get_metadata(hdf5_name, model)        
            df_row = row.tolist() + [
                    code, str(roi), model, '_'.join(meta['dm'])]
            dataframe.append(df_row)

    # And write it out.
    header = score_names + ['sub', 'roi', 'model', 'dm']
    filename = hdf5_name.split('.')[0] +  '_scores.txt'
    fid = open(filename, 'w')
    writer = csv.writer(fid, delimiter='\t')
    writer.writerow(header)
    writer.writerows(dataframe)
    fid.close()


def reformat_model(smobject):
    """Extract many useful results from a statsmodel results
    instance into a dict."""

    if smobject is None:
        return None
    
    tosave = {
        "beta":"params",
        "t":"tvalues",
        "fvalue":"fvalue",
        "p":"pvalues",
        "r":"rsquared",
        "r_adj" : "rsquared_adj",
        "ci":"conf_int",
        "resid":"resid",
        "aic":"aic",
        "bic":"bic",
        "llf":"llf",
        "mse_model":"mse_model",
        "mse_resid":"mse_resid",
        "mse_total":"mse_total"
    }
    
    # Try to get each attr (a value in the dict above)
    # first as function (without args) then as a regular
    # attribute.  If both fail, silently move on.
    results = {}
    for k, v in tosave.items():
        try:
            results[k] = deepcopy(getattr(smobject,v)())
        except TypeError:
            results[k] = deepcopy(getattr(smobject,v))
        except AttributeError:
            continue
    
    return results


def reformat_contrast(stato):
    """Extract all public data from a ContrastResult object"""
    
    if stato is None:
        return None
    
    # Get public attr names
    # And store return in a dict
    names = [
            attr for attr in dir(stato) if not attr.startswith('_')
            ]
            
    stat_results = {}
    for name in names:
        stat_results[name] = getattr(stato, name)
    
    return stat_results

    
def merge_results(name, model, smo, df=None, stato=None, other=None):
    """Merge disparate results objects and dicts into a single dict
    suitable for saving."""

    
    glm_results = reformat_model(smo)
    results = deepcopy(glm_results)
    
    stat_results = reformat_contrast(stato)
    results['tests'] = stat_results
    
    if df is not None:
        results['data'] = df.to_dict('list')    
    else:
        results['data'] = {}
    
    bold, dm = [mo.strip() for mo in model.split('~')]        
    results['data'].update({
            'meta' : {'bold': bold, 'dm': dm}, 'other' : other 
            })
    
    return {name : results}

