import numpy as np
import pandas as pd

import pickle

# load data to pandas dataframe
def load_data(data_dir):
    data = pd.read_csv(data_dir, sep='\t', error_bad_lines=False)
    
    return data

# save object to pickle
def save_pickle(to_save, path):
    with open('{}.p'.format(path), 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Successfully saved to {}.p...'.format(path))

# load pickle object
def load_pickle(path):
    with open('{}.p'.format(path), 'rb') as handle:
        loaded = pickle.load(handle)
    
    print('Successfully loaded {}.p...'.format(path))

    return loaded

# split data into left and right data
def split_to_dict(data, cols):
    data = {'left': data[cols[0]], 'right': data[cols[1]]}

    return data