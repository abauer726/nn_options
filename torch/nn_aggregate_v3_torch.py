#### EDIT THIS TO BE FOR PYTORCH 
# as of 10/17: this page is entirely in tensorflow


# Longstaff Schwartz Algorithm
# Implementation of a aggregate neural network

from payoffs import payoff

## Libraries
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal

def NN_payoff_neo(time_step, stock, model, net, NN, convert_in, convert_out, val, \
                  nn_val = None, nn_dt = None, stop = False, display_time = False):
    '''
    Longstaff Schwartz Algorithm
    Computes the realized payoff at a given time step using either a sequence of 
    neural network objects or an aggregate network.

    Parameters
    ----------
    time_step : Float indicating the initial time step
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    net : String determining the type of neural network
          'seq' : sequence of neural networks
          'agg' : aggregate neural network
    NN : Depending on the value of net 
         - List of Neural Network objects (size M-1)    if net == 'seq'
         - Neural Network object                        if net == 'agg'
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    nn_dt : Float with the initial dt the neural network was trained on
    val : String determining the type of values that will be outputted: 
          'cont' : realized payoffs via continuation values
          'time' : timing values
    nn_val : String determining the type of output the aggregate network returns:
             'cont' : realized payoffs via continuation values
             'time' : timing values
             The default is None.
    stop : Boolean asserting whether to display stopping times. The default is False.
    display_time : Boolean asserting whether to display the time spent per step. 
                   The default is False.

    Returns
    -------
    r_pay : Array of realized payoff (size N)
    t_val : Array of timing values (size N)
    stopT : List of stopping times (size N)
    '''
    if (val != 'cont') and (val != 'time'):
        raise TypeError('Output type: choose between\n'+\
                        ' - cont: continuation value'+\
                        ' - time: timing value')
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
    
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    
    # Initializing stopping times
    if stop:
        stopT = np.repeat(nSteps*model['dt'], nSims) 
    
    # Transforming the time step into an integer
    t_step = int(time_step/model['T']*nSteps) - 1
    if t_step < 0:
        t_step = 0
    if t_step > nSteps-1:
        raise ValueError('Time step is equal to or exceeds maturity.')
    
    # Array of continuation values
    v_cont = payoff(stock[t_step], model) 
    # List of arrays with stopping decisions
    # True means continue; False means stop 
    cont = []
    
    # Forward Loop
    for i in range(t_step, nSteps-1):
        if display_time:
            start_time = time.time()
        
        # Scaling Neural Network inputs
        if net == 'seq':
            aux_ = []
        elif net == 'agg':
            aux_ = [convert_in[0].transform(np.repeat((i+1)/nSteps*model['T'], \
                                nSims).reshape(-1, 1))]
        for j in range(model['dim']):
            aux_.append(convert_in[j+1].transform(stock[i].transpose()[j].reshape(-1, 1)))
        input_scaled = np.stack(aux_).transpose()[0]
        
        # Predicting continuation values 
        # Scaling Neural Network outputs
        if net == 'seq':
            pred = NN[i].predict(input_scaled)
            prediction = np.ravel(convert_out.inverse_transform(pred))
            q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
            q_hat = np.exp(-model['r']*nn_dt)*q_hat
        elif net == 'agg':
            pred = NN.predict(input_scaled)
            prediction = np.ravel(convert_out.inverse_transform(pred))
            if nn_val == 'cont':
                q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
                q_hat = np.exp(-model['r']*nn_dt)*q_hat
            elif nn_val == 'time':
                t_hat = prediction
        imm_pay = payoff(stock[i], model)
        
        
        # Updating the stopping decision
        if i == t_step:
            if nn_val == 'cont':
                logic = np.logical_or(q_hat > imm_pay, imm_pay == 0)
            elif nn_val == 'time':
                logic = np.logical_or(t_hat > 0, imm_pay == 0)
        else:
            if nn_val == 'cont':
                logic = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), cont[-1])
            elif nn_val == 'time':
                logic = np.logical_and(np.logical_or(t_hat > 0, imm_pay == 0), cont[-1])
        cont.append(logic)
        
        # Perform stopping 
        v_cont = np.exp(model['r']*model['dt'])*v_cont
        for k in range(nSims):
            if i == t_step:
                if cont[-1][k] == False:
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['T']/nSteps
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['T']/nSteps
        
        # Displaying Time per Step
        if display_time:
            print('Step:', np.round((i+1)/nSteps*model['T'], 2),'Time:', \
                           np.round(time.time()-start_time, 2), 'sec')
    
    # Computing the terminal payoff
    imm_pay = payoff(stock[-1], model)
    v_cont = np.exp(model['r']*model['dt'])*v_cont
    for k in range(nSims):
        if cont[-1][k] == True:
            v_cont[k] = imm_pay[k]
    
    # Realized payoffs
    r_pay = np.exp(-model['r']*model['dt']*(nSteps-t_step))*v_cont
    if stop:
        if val == 'cont':
            return (r_pay, stopT)
        elif val == 'time':
            # Timing Value
            t_val = r_pay - payoff(stock[t_step], model) 
            return (t_val, stopT)
    else:
        if val == 'cont':
            return r_pay
        elif val == 'time':
            # Timing Value
            t_val = r_pay - payoff(stock[t_step], model) 
            return t_val
        
def NN_payoff_v3(time_step, stock, model, net, NN, convert_in, convert_out, val, \
                  nn_val = None, nn_dt = None, stop = False, display_time = False):
    '''
    Longstaff Schwartz Algorithm
    Computes the realized payoff at a given time step using either a sequence of 
    neural network objects or an aggregate network.

    Parameters
    ----------
    time_step : Float indicating the initial time step
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    net : String determining the type of neural network
          'seq' : sequence of neural networks
          'agg' : aggregate neural network
    NN : Depending on the value of net 
         - List of Neural Network objects (size M-1)    if net == 'seq'
         - Neural Network object                        if net == 'agg'
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    nn_dt : Float with the initial dt the neural network was trained on
    val : String determining the type of values that will be outputted: 
          'cont' : realized payoffs via continuation values
          'time' : timing values
    nn_val : String determining the type of output the aggregate network returns:
             'cont' : realized payoffs via continuation values
             'time' : timing values
             The default is None.
    stop : Boolean asserting whether to display stopping times. The default is False.
    display_time : Boolean asserting whether to display the time spent per step. 
                   The default is False.

    Returns
    -------
    r_pay : Array of realized payoff (size N)
    t_val : Array of timing values (size N)
    stopT : List of stopping times (size N)
    '''
    if (val != 'cont') and (val != 'time'):
        raise TypeError('Output type: choose between\n'+\
                        ' - cont: continuation value'+\
                        ' - time: timing value')
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
    
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    
    # Initializing stopping times
    if stop:
        stopT = np.repeat(nSteps*model['dt'], nSims) 
    
    # Transforming the time step into an integer
    t_step = int(time_step/model['T']*nSteps)
    if t_step < 0:
        t_step = 0
    if t_step > nSteps-1:
        raise ValueError('Time step is equal to or exceeds maturity.')
    
    # Array of continuation values
    v_cont = payoff(stock[0], model) 
    # List of arrays with stopping decisions
    # True means continue; False means stop 
    cont = []
    
    # Forward Loop
    for i in range(t_step, nSteps-1):
        if display_time:
            start_time = time.time()
        
        # Scaling Neural Network inputs
        if net == 'seq':
            aux_ = []
        elif net == 'agg':
            aux_ = [convert_in[0].transform(np.repeat((i+1)/nSteps*model['T'], \
                                nSims).reshape(-1, 1))]
        for j in range(model['dim']):
            aux_.append(convert_in[j+1].transform(stock[i-t_step].transpose()[j].reshape(-1, 1)))
        input_scaled = np.stack(aux_).transpose()[0]
        
        # Predicting continuation values 
        # Scaling Neural Network outputs
        if net == 'seq':
            pred = NN[i].predict(input_scaled)
            prediction = np.ravel(convert_out.inverse_transform(pred))
            q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
            q_hat = np.exp(-model['r']*nn_dt)*q_hat
        elif net == 'agg':
            pred = NN.predict(input_scaled)
            prediction = np.ravel(convert_out.inverse_transform(pred))
            if nn_val == 'cont':
                q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
                q_hat = np.exp(-model['r']*nn_dt)*q_hat
            elif nn_val == 'time':
                t_hat = prediction
        imm_pay = payoff(stock[i-t_step], model)
        
        
        # Updating the stopping decision
        if i == t_step:
            if nn_val == 'cont':
                logic = np.logical_or(q_hat > imm_pay, imm_pay == 0)
            elif nn_val == 'time':
                logic = np.logical_or(t_hat > 0, imm_pay == 0)
        else:
            if nn_val == 'cont':
                logic = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), cont[-1])
            elif nn_val == 'time':
                logic = np.logical_and(np.logical_or(t_hat > 0, imm_pay == 0), cont[-1])
        cont.append(logic)
        
        # Perform stopping 
        v_cont = np.exp(model['r']*model['dt'])*v_cont
        for k in range(nSims):
            if i == t_step:
                if cont[-1][k] == False:
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['T']/nSteps
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['T']/nSteps
        
        # Displaying Time per Step
        if display_time:
            print('Step:', np.round((i+1)/nSteps*model['T'], 2),'Time:', \
                           np.round(time.time()-start_time, 2), 'sec')
    
    # Computing the terminal payoff
    imm_pay = payoff(stock[-1], model)
    v_cont = np.exp(model['r']*model['dt'])*v_cont
    for k in range(nSims):
        if cont[-1][k] == True:
            v_cont[k] = imm_pay[k]
    
    # Realized payoffs
    r_pay = np.exp(-model['r']*model['dt']*(nSteps-t_step))*v_cont
    if stop:
        if val == 'cont':
            return (r_pay, stopT)
        elif val == 'time':
            # Timing Value
            t_val = r_pay - payoff(stock[0], model) 
            return (t_val, stopT)
    else:
        if val == 'cont':
            return r_pay
        elif val == 'time':
            # Timing Value
            t_val = r_pay - payoff(stock[0], model) 
            return t_val
        
def NN_aggregate_neo(model, NN, convert_in, convert_out, nn_val, data = False, 
                     x = None, y = None, stock = None, otm = False, node_num = 16, 
                     epoch_num = 50, batch_num = 64, actfct = 'elu', 
                     initializer = TruncatedNormal(mean = 0.0, stddev = 0.05), 
                     optim = 'adam', lossfct = 'mean_squared_error', display_time = False):
    '''
    Builds an aggreagte neural network from a sequence of neural network objects 

    Parameters
    ----------
    model : Dictionary containing all the parameters of the stock and contract
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    nn_val : String giving the type of output values the neural network returns:
             'cont' : realized payoffs via continuation values
             'time' : timing values
    data : Boolean asserting if the training data should be stored and returned. 
           The default is False.
    x : Array of input data used in training the aggregate neural network
    y : Array of output data used in training the aggregate neural network
    otm : Boolean asserting whether to use all the stock paths (including 
          out-of-the-money paths) to train the sequence of neural networks. 
          The default is False.
    node_num : Integer with the number of nodes. The default is 16.
    epoch_num : Integer with the number of epochs. The default is 50.
    batch_num : Integer with the number of batches. The default is 64.
    actfct : String with the activation function. The default is 'elu'.
    initializer : Keras initializer. The default is set to 
                  TruncatedNormal(mean = 0.0, stddev = 0.05).
    optim : Keras optimizer. The default is 'adam'.
    lossfct : String stating the type of the loss function. The default is 
              'mean_squared_error'.
    display_time : Boolean asserting whether to display the time spent per step. 
                   The default is False.

    Returns
    -------
    NNagg : Aggregate Neural Network object
    '''
    if (nn_val != 'cont') and (nn_val != 'time'):
        raise TypeError('Choose between continuation or timing value.')
    if display_time:
        time_all = time.time()
    nn_dim_agg = model['dim'] + 1    # Dimension of the Aggregate Neural Network
    if data == False:
        nSims = len(stock[0])               # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
             
    if data == False:
        # Selecting In-the-Money paths for the Aggregate Neural Network training
        itm = [] 
        x_agg = [] 
        y_agg = []
        for i in range(0, nSteps-1):
            if display_time:
                start_time = time.time()
            itm.append([])
        
            # Computing the timing values on each path with a sequence of networks
            if nn_val == 'cont':
                q_val = NN_payoff_neo((i+1)/nSteps*model['T'], stock, model, 'seq', \
                                  NN, convert_in, convert_out, val = 'cont')
            elif nn_val == 'time':
                t_val = NN_payoff_neo((i+1)/nSteps*model['T'], stock, model, 'seq', \
                                  NN, convert_in, convert_out,  val = 'time')
            if otm:
                x = stock[i]
                if nn_val == 'cont':
                    y = q_val
                elif nn_val == 'time':
                    y = t_val
            
                x_agg.append(np.hstack((np.repeat((i+1)*model['T']/nSteps, \
                                        nSims).reshape(nSims,1), x)))
                y_agg.append(y)
            else:
                for k in range(nSims):
                    if payoff(stock[i][k], model) > 0:
                        if nn_val == 'cont':
                            itm[i].append([stock[i][k], q_val[k]])
                        elif nn_val == 'time':
                            itm[i].append([stock[i][k], t_val[k]])
                
                x = np.stack(np.array(itm[i], dtype=object).transpose()[0])
                y = np.array(itm[i], dtype=object).transpose()[1]
        
                x_agg.append(np.hstack((np.repeat((i+1)*model['T']/nSteps, \
                                        len(itm[i])).reshape(len(itm[i]),1), x)))
                y_agg.append(y)
        
            # Displaying Time per Step
            if display_time:
                print('Step:',np.round((i+1)/nSteps*model['T'], 2),'Time:', \
                      np.round(time.time()-start_time, 2), 'sec')
    
        # Training data for the aggregate network
        x = np.concatenate(x_agg, axis = 0)
        y = np.concatenate(y_agg, axis = 0)
    
    # Random suhffling of data points
    merge = np.append(x.transpose(), y.transpose()).reshape(model['dim']+2, len(y)).transpose()
    np.random.shuffle(merge)
    x = merge.transpose()[:-1].transpose()
    y = merge.transpose()[-1].reshape(-1, 1)
    
    # Scaling neural network inputs and outputs    
    input_scaled = []
    for j in range(nn_dim_agg):
            input_train = x.transpose()[j].reshape((-1, 1))
            input_scaled.append(convert_in[j].transform(input_train))
    nn_input = np.stack(input_scaled).transpose()[0]
    nn_output = convert_out.transform(y.reshape(-1,1))
    
    # Defining and training the aggeegate neural network
    
    # Try LeakyReLU
    # Try bias 0 (bias can change)
    # Xaiver golrot initialization for weights and biases
    NNagg = Sequential()    
    NNagg.add(Dense(node_num, input_shape = (nn_dim_agg,), activation = actfct, \
                       kernel_initializer = initializer, bias_initializer = initializer))            
    NNagg.add(Dense(node_num, activation = actfct, kernel_initializer = initializer, \
                       bias_initializer = initializer))
    NNagg.add(Dense(node_num, activation = actfct, kernel_initializer = initializer, \
                       bias_initializer = initializer))
    NNagg.add(Dense(1, activation = None, kernel_initializer = initializer, \
                       bias_initializer = initializer)) 
    NNagg.compile(optimizer = optim, loss = lossfct)
    NNagg.fit(nn_input, nn_output, epochs = epoch_num, batch_size = batch_num, verbose = 0)   
    if display_time:
        print('Training time:', np.round(time.time()-time_all, 2), 'sec')
    return NNagg
