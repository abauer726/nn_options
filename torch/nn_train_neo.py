# Longstaff Schwartz Algorithm
# Implementation of the LSM with a sequence of neural networks

from payoffs import payoff

## Libraries
import time
import numpy as np
import tensorflow as tf
tf.autograph.experimental.do_not_convert
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal


def NN_seq_train_neo(stock, model, convert_in, convert_out, theta = 'average',
                     otm = False, data = False, val = None, node_num = 16, epoch_num = 50, 
                     batch_num = 64, actfct = 'elu', initializer = TruncatedNormal(mean = 0.0, stddev = 0.05),
                     optim = 'adam', lossfct = 'mean_squared_error', display_time = False):
    '''
    Longstaff Schwartz Algorithm
    Implementation of the LSM using a sequence of neural network objects. Training 
    is performed along a fixed set of paths.
    
    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects. 
    theta : Initialization policy of the weights and the biases of the neural 
            network objects. The first noural netowrk is initialized with the 
            keras 'initializer'. 
            If theta = 'average', then the weights and biases from subsequent 
            networks are initialized with the averges of the weights and biases 
            from the previous steps. 
            If theta = 'previous', then the weights and biases from subsequent 
            networks are initialized with the the weights and biases from the 
            previous step. 
            If theta = 'random', then the weights and biases of all the neural
            networks are initialized with the keras 'initializer'
            The default is 'average'.
    otm : Boolean asserting whether to use all the stock paths (including 
          out-of-the-money paths) to train the sequence of neural networks. 
          The default is False.
    data : Boolean asserting if the training data should be stored and returned. 
           The default is False.
    val : Selects the data values that will be outputted if data == True: 
          'cont' : realized payoffs via continuation values
          'time' : timing values
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
    NN : List of neural network objects (size M-1)
    x : Input data used to train the sequence of neural network objects
    y : Output data used to train the sequence of neural network objects
    '''
    
    tf.autograph.experimental.do_not_convert(
        func=None
    )
    
    if data:
        if val == None:
            raise TypeError('The type of output values has not been selected!')
        if (val != 'cont') and (val != 'time'):
            raise TypeError('Choose between continuation or timing value!')
    nn_dim = model['dim']                   # Dimension of the neural network
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    q = payoff(stock[-1], model)      # List of continuation values
    NN = []             # List of neural network objects 
    
    # Stores training data
    if data:
        x_agg = []
        y_agg = []
    # Backward Loop
    for i in reversed(range(0,nSteps-1)):
        if display_time:
            start_time = time.time()
        
        if otm:
            # Selecting all paths for training
            x = stock[i]
            y = q
            if data:
                x_agg.append(np.hstack((np.repeat((i+1)*model['T']/nSteps, \
                                    nSims).reshape(nSims,1), x)))
                if val == 'cont':
                    y_agg.append(y)
                elif val == 'time':
                    y_agg.append(y-payoff(x, model)) 
        else:
            # Selecting In-the-Money paths for training
            itm = [] 
            for k in range(nSims):
                if payoff(stock[i][k], model) > 0:
                    itm.append([stock[i][k], q[k]])
            x = np.stack(np.array(itm, dtype=object).transpose()[0])
            y = np.array(itm, dtype=object).transpose()[1]
            
            if data:
                x_agg.append(np.hstack((np.repeat((i+1)*model['T']/nSteps, \
                                len(itm)).reshape((len(itm),1)), x)))
                if val == 'cont':
                    y_agg.append(y)
                elif val == 'time':
                    y_agg.append(y-payoff(x, model)) 
        
        # Scaling neural network inputs       
        input_scaled = []
        for j in range(model['dim']):
            input_train = x.transpose()[j].reshape(-1, 1)
            input_scaled.append(convert_in[j+1].transform(input_train))
        nn_input = np.stack(input_scaled).transpose()[0]
        nn_output = convert_out.transform(y.reshape(-1,1))
        
        # Defining and training the neural network
        if  i == nSteps-2:
            NNet_seq = Sequential()    
            NNet_seq.add(Dense(node_num, input_shape = (nn_dim,), activation = actfct,
                         kernel_initializer = initializer, bias_initializer = initializer))            
            NNet_seq.add(Dense(node_num, activation = actfct,
                         kernel_initializer = initializer, bias_initializer = initializer))
            NNet_seq.add(Dense(node_num, activation = actfct, 
                         kernel_initializer = initializer, bias_initializer = initializer))
            NNet_seq.add(Dense(1, activation = None, 
                         kernel_initializer = initializer, bias_initializer = initializer))
            NNet_seq.compile(optimizer = optim, loss = lossfct)
            NNet_seq.fit(nn_input, nn_output, epochs = epoch_num, \
                         batch_size = batch_num, verbose = 0)
        else:
            if theta == 'average':
                # Average weights and biases
                w_mean = []
                b_mean = []
                w = np.empty(shape = (len(NN), len(NN[0].layers)), dtype = object)
                b = np.empty(shape = (len(NN), len(NN[0].layers)), dtype = object)
                for n in range(len(NN)):
                    for m in range(len(NN[n].layers)): # Number of layers
                        w[n][m] = NN[n].layers[m].get_weights()[0]
                        b[n][m]= NN[n].layers[m].get_weights()[1]
                for m in range(len(NN[0].layers)):
                    w_mean.append(w.transpose()[m].mean())
                    b_mean.append(b.transpose()[m].mean())
            elif theta == 'previous':
                # Previous weights and biases
                w_mean  = []
                b_mean = []
                for lay in NN[-1].layers:
                    w_mean.append(lay.get_weights()[0])
                    b_mean.append(lay.get_weights()[1])
            elif theta == 'random':
                # Random weights and biases
                w_mean = initializer
                b_mean = initializer
                  
            NNet_seq = Sequential()
            NNet_seq.add(Dense(node_num, input_shape = (nn_dim,), activation = actfct,
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[0]), 
                            bias_initializer = tf.keras.initializers.Constant(b_mean[0])))            
            NNet_seq.add(Dense(node_num, activation = actfct,
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[1]),
                            bias_initializer = tf.keras.initializers.Constant(b_mean[1])))
            NNet_seq.add(Dense(node_num, activation = actfct, 
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[2]), 
                            bias_initializer = tf.keras.initializers.Constant(b_mean[2])))
            NNet_seq.add(Dense(1, activation = None, 
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[3]), 
                            bias_initializer = tf.keras.initializers.Constant(b_mean[3])))
            NNet_seq.compile(optimizer = optim, loss = lossfct)
            NNet_seq.fit(nn_input, nn_output, epochs = epoch_num, \
                                  batch_size = batch_num, verbose = 0)
        
        # Storing neural network objects
        NN.append(NNet_seq)
        
        # Predicting continuation values using the neural network
        aux = []
        for j in range(model['dim']):
            aux.append(convert_in[j+1].transform(stock[i].transpose()[j].reshape(-1, 1)))
        input_scaled_all = np.stack(aux).transpose()[0]
        
        pred = NNet_seq.predict(input_scaled_all)
        prediction = np.ravel(convert_out.inverse_transform(pred))
        
        
        # Computing continuation values
        qhat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
        qhat = np.exp(-model['r']*model['dt'])*qhat
        imm_pay = payoff(stock[i], model)
        
        # Updating the continuation values and stopping times
        for k in range(nSims):
            if (imm_pay[k] > 0) and (qhat[k] <= imm_pay[k]):
                q[k] = imm_pay[k]
            else:
                q[k] = np.exp(-model['r']*model['dt'])*q[k]
        
        # Displaying Time per Step
        if display_time:
            print('Step:', np.round((i+1)*model['T']/nSteps, 2),' Time:', \
                  np.round(time.time()-start_time,2), 'sec')
                
    # Reversing lists 
    NN.reverse()
    if data:
        x = np.concatenate(x_agg, axis = 0)
        y = np.concatenate(y_agg, axis = 0)    
        return (NN, x, y)
    else:
        return NN
