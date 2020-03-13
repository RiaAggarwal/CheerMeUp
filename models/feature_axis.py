import sklearn.linear_model as linear_model
import numpy as np


def feature_axis(z, y, method = 'linear', **kwargs):
    
    model = linear_model.LinearRegression(**kwargs)
    
    if method == 'linear':
        model.fit(z, y)
    elif method == 'tanh':
        def arctanh_clip():
            return np.arctanh(np.clip(y, np.tanh(-3), np.tanh(3)))
        
        model.fit(z, arctanh_clip(y))
    else:
        raise Exception('Method not in ["linear", "tanh"]')
        
    return model.coef_.T
        

def normalize_feature_axis(feature_slope):

    feature_direction = feature_slope / ((feature_slope**2).sum())
    return feature_direction

def disentangle_feature_axis(feature_axis_target, feature_axis_base, yn_base_orthogonalized=False):

    # make sure this funciton works to 1D vector
    if len(feature_axis_target.shape) == 0:
        yn_single_vector_in = True
        feature_axis_target = feature_axis_target[:, None]
    else:
        yn_single_vector_in = False

    # if already othogonalized, skip this step
    if yn_base_orthogonalized:
        feature_axis_base_orthononal = orthogonalize_vectors(feature_axis_base)
    else:
        feature_axis_base_orthononal = feature_axis_base

    # orthogonalize every vector
    feature_axis_decorrelated = feature_axis_target + 0
    num_dim, num_feature_0 = feature_axis_target.shape
    num_dim, num_feature_1 = feature_axis_base_orthononal.shape
    for i in range(num_feature_0):
        for j in range(num_feature_1):
            feature_axis_decorrelated[:, i] = orthogonalize_one_vector(feature_axis_decorrelated[:, i],
                                                                       feature_axis_base_orthononal[:, j])

    # make sure this funciton works to 1D vector
    if yn_single_vector_in:
        result = feature_axis_decorrelated[:, 0]
    else:
        result = feature_axis_decorrelated

    return result