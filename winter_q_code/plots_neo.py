from payoffs import payoff

## Libraries
import time
import numpy as np

def NN_bound_neo(NN, convert_in, convert_out, model, net, nn_val = None, nn_dt = None,  
                 down = 0.75, up = 1.01, inc = 0.05, display_time = False):
    '''
    Computes the bound parameters for 1-dim options 

    Parameters
    ----------
    NN : Depending on the value of net 
         - List of Neural Network objects (size M-1)    if net == 'seq'
         - Neural Network object                        if net == 'agg'
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    model : Dictionary containing all the parameters of the stock and contract
    net : String determining the type of neural network
          'seq' : sequence of neural networks
          'agg' : aggregate neural network
    nn_val : String determining the type of output the aggregate network returns:
             'cont' : realized payoffs via continuation values
             'time' : timing values
             The default is None.
    nn_dt : Float with the initial dt the neural network was trained on
    down : Float scaling parameter for the lower bound of the stock price. 
           The default is 0.75 or 75% of the initial value.
    up : Float scaling parameter for the upper bound of the stock price. 
         The default is 1.01 or 101% of the initial value.
    inc : Float increment for the stock price grid. The default is 0.05.
    display_time : Display time spent per step. The default is False.

    Returns
    -------
    Tuple consisting of bound parameters and timing values
    '''
    if (net != 'seq') and (net != 'agg'):
        raise TypeError('Network type: choose between\n'+\
                        ' - seq: sequence of networks\n' +\
                        ' - agg: aggregate network')
    if net == 'agg':
        if (nn_val == None) or ((nn_val != 'cont') and (nn_val != 'time')):
            raise TypeError('Aggregate network: choose between\n'+\
                            ' - cont: continuation\n' +\
                            ' - time: timing value')
        if nn_dt == None:
            raise ValueError('Provide the initial dt the aggregate network was trained on')
    elif net == 'seq':
        nn_val = 'cont'
        nn_dt = model['dt']
    if display_time:
        start_time = time.time()
    
    nSteps = int(model['T']/model['dt'])
    if model['dim'] != 1:
        raise TypeError('Only works for 1-D options.')
    
    prices = np.arange(model['x0']*down, model['x0']*up, inc)
    n = len(prices)
    steps = np.arange(model['dt'], model['T'], model['dt']) 
    # Immediate payoff of the stock prices
    imm_pay = payoff(np.reshape(prices, (n,1)), model)
    
    # Predicting continuation values and scaling neural network outputs
    q = []    # List of continuation values
    for i in range(nSteps-1):
        if net == 'seq':
            input_scaled = convert_in[1].transform(prices.reshape(-1, 1))
        elif net == 'agg':
            aux_ = [convert_in[0].transform(np.repeat((i+1)*model['T']/nSteps,n).reshape(-1, 1))]
            aux_.append(convert_in[1].transform(prices.reshape(-1, 1)))
            input_scaled = np.stack(aux_).transpose()[0]
        
        if net == 'seq':
            pred = NN[i].predict(input_scaled)
        elif net == 'agg':
            pred = NN.predict(input_scaled)
        prediction = np.ravel(convert_out.inverse_transform(pred))
        if nn_val == 'cont':
            q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
            q_hat = np.exp(-model['r']*nn_dt)*q_hat
            q.append(list(q_hat - imm_pay))
        elif nn_val == 'time':
            t_hat = prediction
            q.append(list(t_hat))
    if display_time:
        print('Time:', np.round(time.time()-start_time,2), 'sec')
    return (steps, prices, np.array(q).transpose())

def NN_contour_neo(t_step, NN, convert_in, convert_out, model, net, nn_val = None,
               nn_dt = None, down = 0.8, up = 1.8, inc = 0.1, display_time = False):
    '''
    Returns contour parameters for 2-dim options at a particular time-step
    from a single neural network object
    
    Parameters
    ----------
    t_step : Integer with the time step 
    NN : Depending on the value of net 
         - List of Neural Network objects (size M-1)    if net == 'seq'
         - Neural Network object                        if net == 'agg'
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    model : Dictionary containing all the parameters of the stock and contract
    net : String determining the type of neural network
          'seq' : sequence of neural networks
          'agg' : aggregate neural network
    nn_val : String determining the type of output the aggregate network returns:
             'cont' : realized payoffs via continuation values
             'time' : timing values
             The default is None.
    nn_dt : Float with the initial dt the neural network was trained on         
    down : Scaling parameter for the lower bound of the map. 
           The default is 0.6 or 60% of the initial price.
    up : Scaling parameter for the upper bound of the map. 
         The default is 1.4 or 140% of the initial price.
    inc : Increment for the stock price grid. The default is 0.1.
    display_time : Display time spent per step. The default is False.

    Returns
    -------
    Tuple consisting of contour parameters and timing values
    '''
    if (net != 'seq') and (net != 'agg'):
        raise TypeError('Network type: choose between\n'+\
                        ' - seq: sequence of networks\n' +\
                        ' - agg: aggregate network')
    if net == 'agg':
        if (nn_val == None) or ((nn_val != 'cont') and (nn_val != 'time')):
            raise TypeError('Aggregate network: choose between\n'+\
                            ' - cont: continuation\n' +\
                            ' - time: timing value')
        if nn_dt == None:
            raise ValueError('Provide the initial dt the aggregate network was trained on')
    elif net == 'seq':
        nn_val = 'cont'
        nn_dt = model['dt']

    nSteps = int(model['T']/model['dt'])
    if net == 'seq':
        if not(np.round(t_step,4) in np.round(np.arange(0, model['T'], model['dt']),4)):
            raise TypeError('Time step is out of range for the model dt.')
        step = int(round(t_step/model['dt'])) - 1   
    if model['dim'] != 2:
        raise TypeError('Only works for 2-dim options.')
    
    if display_time:
        start_time = time.time()    
    prices = []
    for x0 in model['x0']:
        prices.append(np.arange(x0*down, x0*up, inc))
    n = len(prices[0])
    x, y = np.meshgrid(prices[0], prices[1])
    aux = []
    for a, b in zip(x, y):
        aux.append(np.reshape(np.ravel([a,b]), (n,2), order = 'F'))
    prices_ = np.reshape(np.ravel(aux), (n**2, 2), order='C')
    # Immediate payoff of the stock price grid
    imm_pay = payoff(prices_, model)
    # Transforming the input prices for the neural network
    if net == 'seq':
        aux_ = []
        for j in range(model['dim']):
            aux_.append(convert_in[j+1].transform(prices_.transpose()[j].reshape(-1, 1)))
        input_scaled = np.stack(aux_).transpose()[0]
    elif net == 'agg':
        aux_ = [convert_in[0].transform(np.repeat(t_step,n**2).reshape(-1, 1))]
        for j in range(model['dim']):
            aux_.append(convert_in[j+1].transform(prices_.transpose()[j].reshape(-1, 1)))
        input_scaled = np.stack(aux_).transpose()[0]
    
    # Predicting continuation values and scaling neural network outputs
    if net == 'seq':
        pred = NN[step].predict(input_scaled)
    elif net == 'agg':
        pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out.inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
    q_hat = np.exp(-model['r']*nn_dt)*q_hat
    q = q_hat - imm_pay 
    q = np.reshape(q, (n, n))
    if display_time:
        print('Time:', np.round(time.time()-start_time,2), 'sec')
    return (prices[0], prices[1], q)

def NN_time_neo(position, NN, convert_in, convert_out, model, nn_val = None,  
                nn_dt = None, t_inc = 0.01, display_time = False):
    '''
    Computes a tuple of time and timing values for a specified starting position

    Parameters
    ----------
    position : Float with stock price (size model['dim'])
    NN : Neural Network object                        
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    model : Dictionary containing all the parameters of the stock and contract
    nn_val : String determining the type of output the aggregate network returns:
             'cont' : realized payoffs via continuation values
             'time' : timing values
             The default is None.
    nn_dt : Float with the initial dt the neural network was trained on
    t_inc : Float with the time increment of the grid. The default is 0.01.
    display_time : Display time spent per step. The default is False.

    Returns
    -------
    Tuple consisting of a time grid and timing values
    '''
    if (nn_val == None) or ((nn_val != 'cont') and (nn_val != 'time')):
        raise TypeError('Aggregate network: choose between\n'+\
                        ' - cont: continuation\n' +\
                        ' - time: timing value')
    if nn_dt == None:
            raise ValueError('Provide the initial dt the aggregate network was trained on')
    
    if type(position) != type(np.array([])):
        raise TypeError('position - needs to be a numpy array!')
    if display_time:
        start_time = time.time()
    
    steps = np.arange(0, model['T'], t_inc) 
    n = len(steps)
    # Immediate payoff of the stock prices
    imm_pay = payoff(np.tile(position, (n, 1)), model)
    
    # Predicting continuation values and scaling neural network outputs
    q = []    # List of continuation values
    aux_ = [convert_in[0].transform(steps.reshape(-1, 1))]
    for j in range(model['dim']):
        aux_.append(convert_in[j+1].transform(np.repeat(position[j], n).reshape(-1, 1)))
    input_scaled = np.stack(aux_).transpose()[0]
        
    pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out.inverse_transform(pred))
    if nn_val == 'cont':
        q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
        # Discounting under review
        q_hat = np.exp(-model['r']*nn_dt)*q_hat
        q.append(list(q_hat - imm_pay))
    elif nn_val == 'time':
        t_hat = prediction
        q.append(list(t_hat))
    if display_time:
        print('Time:', np.round(time.time()-start_time,2), 'sec')
    return (steps, np.array(q).transpose())