## Libraries
import numpy as np

def payoff(stock_val, model):
    '''
    Generates option payoffs
    
    Parameters
    ----------
    stock_val : matrix of size N x model['dim'] or vector of size model['dim'] 
    model : Dictionary containing all the parameters, including strike, 
            and the payoff function

    Returns
    -------
    A vector of size N with option payoffs if stock_val is a matrix 
    An option payoff if stock_val is a vector of size model['dim']
    '''
    try:
        (nSims, dim) = stock_val.shape
    except ValueError:
        dim = None
    
    
    ## Arithmetic basket Put on average asset price
    ## Put payoff \eqn{(K-mean(x))_+}
    if model['payoff.func'] == 'put.payoff':
        if dim:
            return np.array([model['K']-np.ndarray.mean(stock_val, axis = 1), \
                             np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([model['K'] - np.ndarray.mean(stock_val, axis = 0), \
                             np.zeros(1)], dtype=object).max(axis = 0)
    
    ## Multivariate Min Put
    ## Min Put payoff \eqn{(K-min(x))_+}
    elif model['payoff.func'] == 'mini.put.payoff':
        if dim: 
            return np.array([model['K']-np.ndarray.min(stock_val, axis = 1), \
                     np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([model['K']-np.ndarray.min(stock_val, axis = 0), \
                     np.zeros(1)], dtype=object).max(axis = 0)
        
    ## Arithmetic basket Call on average asset price
    ## Call payoff \eqn{(mean(x)-K)_+}
    elif model['payoff.func'] == 'call.payoff': 
        if dim: 
            return np.array([np.ndarray.mean(stock_val, axis = 1) - model['K'], \
                     np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([np.ndarray.mean(stock_val, axis = 0) - model['K'], \
                     np.zeros(1)], dtype=object).max(axis = 0)
    
    ## Multivariate Max Call
    ## Max Call payoff \eqn{(max(x)-K)_+}
    elif model['payoff.func'] == 'maxi.call.payoff':
        if dim: 
            return np.array([np.ndarray.max(stock_val, axis = 1) - model['K'], \
                     np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([np.ndarray.max(stock_val, axis = 0) - model['K'], \
                     np.zeros(1)], dtype=object).max(axis = 0)