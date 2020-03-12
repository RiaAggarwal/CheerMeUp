import sklearn.linear_model as linear_model
import numpy as np
def feature_axis(z, y, method = 'linear', **kwargs):
    if method == 'linear':
        model = linear_model.LinearRegression(**kwargs)
        model.fit(z, y)
    elif method == 'tanh':
        def arctanh_clip():
            return np.arctanh(np.clip(y, np.tanh(-3), np.tanh(3)))
        