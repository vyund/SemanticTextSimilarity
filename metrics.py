import keras.backend as K

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# calculate Manhattan Distance as defined by Shirkhorshidi, Aghabozorgi, Wah (2015)
def manhattan_dist(subtracted):
    return K.exp(-K.sum(K.abs(subtracted), axis=1, keepdims=True))

# calculate Cosine Similarity as defined by Shirkhorshidi, Aghabozorgi, Wah (2015)
def cosine_sim(left, right):
    left_norm = K.l2_normalize(left, axis=-1)
    right_norm = K.l2_normalize(right, axis=-1)

    return K.exp(K.sum(K.prod([left_norm, right_norm], axis=0), axis=1, keepdims=True))

# calculate Pearson Correlation between truths and predictions
def pearson_corr(y_true, y_pred):
    y_true = np.array(y_true, dtype='float32')
    y_pred = np.ravel(y_pred)

    r, _ = pearsonr(y_true, y_pred)

    return r

# calculate Spearman Correlation between truth and predictions
def spearman_corr(y_true, y_pred):
    y_true = np.array(y_true, dtype='float32')
    y_pred = np.ravel(y_pred)

    p, _ = spearmanr(y_true, y_pred)

    return p
