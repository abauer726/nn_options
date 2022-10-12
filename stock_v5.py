# Simulating stock prices

from payoffs import payoff

## Libraries
import numpy as np
import random
from scipy.stats import qmc

def sim_gbm(x0, model):
    '''
    Simulate paths of Geometric Brownian Motion with constant parameters
    Simulate from \eqn{p(X_t|X_{t-1})}
    
    Parameters
    ----------
    x0 : Starting values (matrix of size N x model['dim'])
    model : Dictionary containing all the parameters, including volatility,
            interest rate, and continuous dividend yield

    Returns
    -------
    A matrix of same dimensions as x0
    '''
    length = len(x0)
    newX = []
    dt = model['dt']
    for j in range(model['dim']): # indep coordinates
       newX.append(x0[:,j]*np.exp(np.random.normal(loc = 0, scale = 1, size = length)*
                                 model['sigma'][j]*np.sqrt(dt) +
            (model['r'] - model['div'][j] - model['sigma'][j]**2/2)*dt))
    return np.reshape(np.ravel(np.array(newX)), (length, model['dim']), order='F')

def QMC_neo(n, t_step, factor, quasimc, model):
    '''
    Space-filling on a specified hyper-rectangular
    
    Parameters
    ----------
    n : Integer
    factor : Float 
    quasimc : String recording the type
              'Sobol'
    model : Dictionary containing all the parameters, including volatility, 
            interest rate, and continuous dividend yield
        
    Returns
    -------
    Array 
    '''
    ### DELETE Plot
    ### plt.scatter(igen.transpose()[0], igen.transpose()[1])
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    t_step = int(t_step/model['T']*nSteps)
    if t_step > nSteps:
        raise ValueError('Time step is equal to or exceeds maturity.')
    if quasimc == 'Sobol':
        sampler = qmc.Sobol(d=model['dim'], scramble=False)
    elif quasimc == 'Halton':
        sampler = qmc.Halton(d=model['dim'], scramble=False)
    elif quasimc == 'LatinHypercube':
        sampler = qmc.LatinHypercube(d=model['dim'])
    # sampler.fast_forward(n*t_step)
    input_range = model['x0']*factor
    sampler.fast_forward(random.randint(0, n**3))
    sample = model['x0'] - input_range + 2*input_range*sampler.random(n)
    # Select only in-the-money
    return np.delete(sample, np.where(payoff(sample, model) == 0), axis = 0)

def stock_sim(nSims, model, start = None):
    '''
    Simulates stock paths using Geometric Brownian Motion
    
    Parameters
    ----------
    nSims : Integer with the number of simulations -- N
    model : Dictionary containing all the parameters, including volatility, 
            interest rate, and continuous dividend yield
    start : Array with starting values of the stock. The default is None corresponding 
            to initializing the N x model['x0'].

    Returns
    -------
    An arrayof shape M x N x model['dim']
    where M is the number of steps model['T']/model['dt']
    '''
    if start is None:
        start = np.reshape(np.repeat(model['x0'], nSims), (nSims, model['dim']))
    if start.shape != (nSims, model['dim']):
        start = np.reshape(start, (nSims, model['dim']))
    nSteps = int(model['T']/model['dt'])
    test_j = []
    test_j.append(sim_gbm(start, model))
    for i in range(1,nSteps):
        test_j.append(sim_gbm(test_j[i-1], model))
    return np.reshape(np.ravel(test_j), (nSteps, nSims, model['dim']))

def stock_sim_neo(nSims, t_point, x_point, model):
    '''
    Simulates stock paths using Geometric Brownian Motion
    
    Parameters
    ----------
    nSims : Integer with the number of simulations -- N
    t_point : Float with the starting time.
    x_point : Array with starting values of the stock. ()
    model : Dictionary containing all the parameters, including volatility, 
            interest rate, and continuous dividend yield
    Returns
    -------
    Array of shape M x N x model['dim']
    where M is the number of steps model['T']/model['dt']
    '''
    if type(x_point) != type(np.array([])):
        raise TypeError('x_points need to be a Numpy Array.')
    if len(x_point) != model['dim']:
        raise ValueError('x_points  does not match the dimension of the model.')
    if not(t_point in np.round(np.arange(0, model['T'], model['dt']), 5)):
        raise ValueError('t_point is not in the proper range.')
    start = np.reshape(np.repeat(x_point, nSims), (nSims, model['dim']), order='F')
    nSteps = int(model['T']/model['dt'])
    step = int(np.round(t_point/model['dt']))
    test_j = []
    test_j.append(sim_gbm(start, model))
    for i in range(1,nSteps-step):
        test_j.append(sim_gbm(test_j[i-1], model))
    return np.reshape(np.ravel(test_j), (nSteps-step, nSims, model['dim']))

def stock_sim_v3(t_point, start, model):
    '''
    Simulates stock paths using Geometric Brownian Motion
    
    Parameters
    ----------
    t_point : Float with the starting time.
    start : Array with starting values of the stock. (size N x model['dim'])
    model : Dictionary containing all the parameters, including volatility, 
            interest rate, and continuous dividend yield
    Returns
    -------
    Array of shape M x N x model['dim']
    where M is the number of steps model['T']/model['dt']
    '''
    if type(start) != type(np.array([])):
        raise TypeError('x_points needs to be a Numpy Array.')
    if start.shape[1] != model['dim']:
        raise ValueError('x_points  does not match the dimension of the model.')
    if not(t_point in np.round(np.arange(0, model['T'], model['dt']), 5)):
        raise ValueError('t_point is not in the proper range.')
    nSteps = int(model['T']/model['dt'])
    step = int(np.round(t_point/model['dt']))
    test_j = []
    test_j.append(sim_gbm(start, model))
    for i in range(1,nSteps-step):
        test_j.append(sim_gbm(test_j[i-1], model))
    return np.reshape(np.ravel(test_j), (nSteps-step, len(start), model['dim']))


def stock_thin(stock, model, dt):
    '''
    Thins the stock paths with a given frequency
    
    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters, including volatility, 
            interest rate, and continuous dividend yield
    dt : Float with time per step.
    
    Returns
    -------
    An array of shape freq x N x model['dim']
    where M is the number of steps model['T']/model['dt']
    '''
    freq = int(model['T']/dt)   # Thinning frequency 
    if freq > len(stock):
        raise ValueError('Selected frequency exceeds the current number of steps.')
    aux = []
    #for i in range(0, len(stock), int(len(stock)/freq)):
    #    aux.append(stock[i])
    for i in range(int(len(stock)/freq)-1, len(stock), int(len(stock)/freq)):
        aux.append(stock[i])
    return np.stack(aux)

