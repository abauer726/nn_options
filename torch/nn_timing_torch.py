from payoffs import payoff

## Libraries
import time
import numpy as np


def NN_timing(t, x, NN, convert_in, convert_out, model, nn_dt = None, 
              display_time = False):
    '''
    Computes the timing values using the aggregate NN for specified starting positions
    Parameters
    ----------
    position : Float with stock price (size model['dim'])
    NN : Neural Network object                        
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    model : Dictionary containing all the parameters of the stock and contract
    nn_dt : Float with the initial dt the neural network was trained on
    display_time : Display time spent per step. The default is False.
    Returns
    -------
    Tuple consisting of a time grid and timing values
    '''
    if nn_dt == None:
            raise ValueError('Provide the initial dt the aggregate network was trained on')
    
    if type(x) != type(np.array([])):
        raise TypeError('position - needs to be a numpy array!')
    if display_time:
        start_time = time.time()
    
    n = len(x)
    # Immediate payoff of the stock prices
    imm_pay = payoff(x, model)
    t = round(t/model['dt'])*model['dt']    # Dealing with rounding errors
    
    # Predicting continuation values and scaling neural network outputs
    aux_ = [convert_in[0].transform(np.repeat(t, n).reshape(-1, 1))]
    for j in range(model['dim']):
        aux_.append(convert_in[j+1].transform(x.transpose()[j].reshape(-1, 1)))
    input_scaled = np.stack(aux_).transpose()[0]
        
    pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out.inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
    # Discounting under review
    q_hat = np.exp(-model['r']*nn_dt)*q_hat
    
    if display_time:
        print('Time:', np.round(time.time()-start_time,2), 'sec')
    # Returns the timing values
    return q_hat - imm_pay

def NN_timing_neo(t_step, x_data, NN, convert_in, convert_out, model, display_time = False):
    '''
    Returns the timing values at a particular time-step for a single neural network object
    
    This is an old version of the code
    
    Parameters
    ----------
    t_step : Integer with the time step 
    NN : Depending on the value of net 
         - List of Neural Network objects (size M-1)    if net == 'seq'
         - Neural Network object                        if net == 'agg'
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    model : Dictionary containing all the parameters of the stock and contract
    display_time : Display time spent per step. The default is False.
    Returns
    -------
    q : Timing values computed for the given data input
    '''
    if display_time:
        start_time = time.time()    
    
    # Select data for time step t_step
    x_filter = []
    for x in x_data:
        if x[0] == t_step:
            x_filter.append(x[1:])
    x_filter = np.asarray(x_filter)
    n = len(x_filter)
    
    # Immediate payoff of the stock price grid
    imm_pay = payoff(x_filter, model)
    # Transforming the filtered data for the neural network
    aux_ = [convert_in[0].transform(np.repeat(t_step, n).reshape(-1, 1))]
    for j in range(model['dim']):
        aux_.append(convert_in[j+1].transform(x_filter.transpose()[j].reshape(-1, 1)))
    input_scaled = np.stack(aux_).transpose()[0]
    
    # Predicting continuation values and scaling neural network outputs
    pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out.inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
    q_hat = np.exp(-model['r']*model['dt'])*q_hat
    q = q_hat - imm_pay 
    if display_time:
        print('Time:', np.round(time.time()-start_time,2), 'sec')
    return q
