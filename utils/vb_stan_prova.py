from collections import OrderedDict
import numpy as np
from scipy.special import comb, loggamma

def vb_extract_sample(results):
    # Extract relevant data from the list (adjusting the indices)
    param_specs = results[5]  # sp_mean, this is assumed to be the parameter names
    samples = results[6]      # sp_var, this is assumed to be the parameter samples
    n = len(samples)          # Number of samples for each parameter

    # First pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) for i in splt[1][:-1].split(',')]  # Handle multi-dimensional parameters
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # Create arrays for each parameter
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape))) for name, shape in param_shapes.items()])

    # Second pass, fill arrays with the parameter samples
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # Adjust for 1-based indexing in STAN
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params




def convert_params(mu, phi):
    """ 
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float 
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    p = mu / (mu + phi)
    r = phi
    return r, p



def neg_binomial_2_rng(mu, phi):
    """Generate a sample from neg_binomial_2(mu, phi).

    This function is defined here in Python rather than in Stan because Stan
    reports overflow errors.

    $E(Y) = mu$
    $Var(Y) = mu + mu^2 / phi

    This function will work fine with arrays.
    """
    tem = np.random.gamma(phi, mu / phi)
    if(tem > 1e15):
        return -1
    else:
        return np.random.poisson(tem)



def neg_binomial_2_lpmf2(y, mu, phi):
    '''
    Compute negative binomial log probability mass function 
    '''
    return np.log(comb(y+phi-1,y)) + y*np.log(mu/(mu+phi)) + phi*np.log(phi/(mu+phi))


def neg_binomial_2_lpmf(y, mu, phi):
    '''
    Compute negative binomial log probability mass function 
    '''
    r, p = convert_params(mu, phi)
    
    return loggamma(r+y) - loggamma(y+1) - loggamma(r) + y*np.log(p) + r*np.log(1-p) 
    
    #return np.log(comb(y+phi-1,y)) + y*np.log(mu/(mu+phi)) + phi*np.log(phi/(mu+phi))

def vb_extract_mean(results):
    # Extract relevant data from the list (adjusting the indices)
    param_specs = results[5]  # sp_mean, assumed to contain parameter names (list of names)
    mean_est = results[6]     # sp_var, assumed to contain parameter estimates (list of estimates)
    
    # First pass, calculate the shape of each parameter
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) for i in splt[1][:-1].split(',')]  # Handle multi-dimensional parameters
        else:
            idxs = (1,)  # Default to scalar if no index exists
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))
    
    # Create arrays for each parameter's mean estimate
    params = OrderedDict([(name, np.nan * np.empty(shape)) for name, shape in param_shapes.items()])

    # Second pass, fill arrays with the mean estimates
    for param_spec, est in zip(param_specs, mean_est):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # Adjust for 1-based indexing in STAN
        else:
            idxs = (0,)  # Default to scalar if no index exists
        params[name][tuple(idxs)] = est
    
    return params
