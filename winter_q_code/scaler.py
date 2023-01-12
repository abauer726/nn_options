# Computing scalers

from payoffs import payoff

## Libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scaler(stock, model, otm = False):
    '''
    Computes the scalers for the sequence of neural networks and the aggregate 
    neural network

    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    otm : Boolean asserting whether to use all the stock paths (including 
          out-of-the-money paths) to train the sequence of neural networks. 
          The default is False.

    Returns
    -------
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects 
    '''
    nSims = len(stock[0])                   # Number of Simulations
    q = payoff(stock[-1], model)            # Payoff at last step
    
    if otm:
        # Selecting all paths
        x = stock[-1]
        y = q
    else:
        # Selecting In-the-Money paths
        itm = [] 
        for k in range(nSims):
            if payoff(stock[-1][k], model) > 0:
                itm.append([stock[-1][k], q[k]])
        
        x = np.stack(np.array(itm, dtype=object).transpose()[0])
        y = np.array(itm, dtype=object).transpose()[1]
    
    # Scaling inputs 
    dim_scaler = MinMaxScaler(feature_range = (0,1)) # Try (-1, 1)
    dim_train = np.arange(0, model['T']+model['dt'], model['dt']).reshape(-1, 1)
    dim_scaler.fit(dim_train)    
    convert_in = [dim_scaler]     
    for j in range(model['dim']):
        input_train = x.transpose()[j].reshape((-1, 1))
        input_scaler = MinMaxScaler(feature_range = (0,1))
        input_scaler.fit(input_train)
        convert_in.append(input_scaler)
    
    # Scaling outputs
    convert_out = MinMaxScaler(feature_range = (0,1))
    convert_out.fit(y.reshape(-1, 1))
    
    return (convert_in, convert_out)