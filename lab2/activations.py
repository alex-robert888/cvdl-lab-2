import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    x = x - np.max(x)
    exp_x = np.exp(x / t)
    sum_exp_x = np.sum(exp_x)
    sm_x = exp_x / sum_exp_x
    return sm_x
