## Stage 3 Updating

from model_update import model_update
from stock_v9 import sim_gbm
from stock_v9 import stock_sim
from stock_v9 import stock_sample_v9
from payoffs import payoff
from nn_aggregate_v4 import NN_payoff_mt


## Libraries
import copy
import random
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def NN_loop_v28(loops, loop_size, model, NN, convert_in, convert_out, target, \
                stock_check = None, epoch_num = 5, batch_num = 64, \
                opt_param = None, lossfct = None, display_time = False):
    '''
    Reinforcement Learning
    Updates the aggreagte neural network in learning loops

    Parameters
    ----------
    loops : Integer with the number of self loops used in retraining the network
    loop_size : Integer with the number of simulations per loop
    model : Dictionary containing all the parameters of the stock and contract
    NN : Aggregate Neural Network object
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling object
    check : Number of loops at which the updated aggregate network is tested. 
            The default is None which corresponds to no testing.
    stock_check : Array of simulated stock paths used in testing the updated
                  aggregate network. The default is None which corresponds to
                  testing on a random set of paths.
    epoch_num : Number of epochs for retraining the aggregate network. The default is 50.
    batch_num : Number of batches for retraining the aggregate network. The default is 64.
    display_time : Display time spent per loop. The default is False.

    Returns
    -------
    NN : Updated Aggregate Neural Network object
    '''
    aux_dt = copy.deepcopy(model['dt'])
    model_update(model, dt = target)
    
    if lossfct == None:
        lossfct = 'mean_squared_error'
    if opt_param == None:
        lr, b1, b2 = 0.001, 0.95, 0.999
    else:
        lr, b1, b2 = opt_param[0], opt_param[1], opt_param[2]
    if type(stock_check) != type(np.array([])):
        stock_check = stock_sim(loop_size, model)
       
    sample = stock_sample_v9(loop_size, model, NN, convert_in, convert_out)
    
    # Computing the option price for the validation paths
    r_check, stop_check = NN_payoff_mt(0, stock_check, model, 'agg', NN, convert_in, \
            convert_out, stop = True)
    max_check = np.mean(r_check)
    
    if display_time:
        print('Stage 2 Price:', np.round(max_check, 4))
    
    # Reinforcemnt Learning
    price_data = [max_check]    # List of option prices
    loop_max = None             # Loop at which the maximum option price is reached
    count_update = 0
    for loop in range(loops):
        if display_time:
            loop_time = time.time()
        
        # Update sample 
        if loop_max == loop - 1 or count_update >= 5:
            count_update = 0
            sample = stock_sample_v9(loop_size, model, NN, convert_in, convert_out)
        else:
            count_update += 1
            
        if sample.shape[0] > loop_size:
            idx = random.sample(range(sample.shape[0]), loop_size)
            x = sample[idx, :]
        else:
            x = sample
            
        '''
        plt.scatter(sample[40:130].transpose()[1], sample[40:130].transpose()[2], s = 0.15)
        plt.scatter(sample[4000:4330].transpose()[1], sample[4000:4330].transpose()[2], s = 0.15)
        plt.savefig('Pos-TV-'+str(loop+1)+'.png', dpi=1000)
        plt.clf()
        '''
        
        (y, sT) = NN_loop_pay_v11(copy.deepcopy(x), model, NN, convert_in, convert_out)
        
        # y_aux = y[y != 0]
        # x = x[y != 0]
        # y = y_aux
        # y = payoff(x.transpose()[1:].transpose(), model).astype(float)
        # plt.hist(y - payoff(x.transpose()[1:].transpose(), model))
        
        '''
        binno = np.arange(min(sT), model['T'], model['dt'])
        # np.round(np.arange(model['dt'], model['T'], model['dt']), 4)
        plt.hist(sT, bins = binno[:-1], color='white', edgecolor='darkblue')
        plt.savefig('V27-Loop-'+str(loop+1)+'.png', dpi=1000)
        plt.clf()
        '''
        
        # Scaling neural network inputs and outputs   
        input_train_scaled_agg = []
        for j in range(model['dim']+1):
            input_train = x.transpose()[j].reshape(-1, 1)
            input_train_scaled_agg.append(convert_in[j].transform(input_train))
        x_ = np.stack(input_train_scaled_agg).transpose()[0]
        y_ = convert_out.transform(y.reshape(-1,1))
        
        # Defining and training the aggregate neural network
        opt = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = b1, beta_2 = b2)
        NN.compile(optimizer = opt, loss = lossfct)
        NN.fit(x_, y_, epochs = epoch_num, batch_size = batch_num, verbose = 0)
        
        # Computing the option price for the validation paths
        r_check, stop_check = NN_payoff_mt(0, stock_check, model, 'agg', NN, \
                    convert_in, convert_out, stop = True)
        
        r_curr = np.mean(r_check)
        if r_curr >= price_data[-1]:
            lr = np.round(lr/1.7, 10)
        else:
            lr = np.round(lr*1.2, 10)
        price_data.append(r_curr)
        
        # Saving the NN that yields the highest option price
        if r_curr > max_check:
            max_check = r_curr
            NN_max = tf.keras.models.clone_model(NN)
            NN_max.build((None, model['dim'] + 1)) 
            NN_max.compile(optimizer = opt, loss = lossfct)
            NN_max.set_weights(NN.get_weights())
            loop_max = loop
        
        # Stop when the learning rate gets very small
        if lr < 5*10**(-8):
            break
        
        # DELETE
        '''
        bins_ = np.round(np.arange(model['dt'], model['T'], model['dt']), 4)
        plt.hist(np.round(stop_check, 4), bins=bins_, linewidth=1, edgecolor='darkblue', facecolor='white', align='mid')
        plt.title('Histogram Loop: '+str(loop+1)+' Price: '+str(np.round(r_curr, 4)))
        plt.savefig('H-Loop-'+str(loop+1)+'-price-'+str(np.round(r_curr, 4))+'.png', dpi=1000)
        plt.clf()
        '''
        # Displaying Time per Loop
        if display_time:
            print('Loop:', loop+1,'LR:', np.round(lr,7), 'Sample:', sample.shape[0],'Price:', np.round(r_curr, 4),\
                  'Paths', len(y_), 'Time:', np.round(time.time()-loop_time,2), 'sec')    
    model_update(model, dt = aux_dt)
    return NN_max

def NN_loop_pay_v11(inputs, model, NN, convert_in, convert_out, display_time = False):
    '''
    Longstaff Schwartz Algorithm
    Computes the loop payoff at a given time step of an aggregate network.

    Parameters
    ----------
    inputs : Array of shape N x (model['dim']+1) of inputs
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    NN : Aggregate Neural Network object                        
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    stop : Boolean asserting whether to display stopping times. The default is False.
    display_time : Boolean asserting whether to display the time spent per step. 
                   The default is False.

    Returns
    -------
    r_pay : Array of realized payoff (size N)
    stop : Array of stopping times (size N) - may stop at t_step
    '''
    
    nSims = inputs.shape[0]                 # Number of Simulations
    stock = copy.deepcopy(inputs.transpose()[1:].transpose())
    times = copy.deepcopy(inputs.transpose()[0].transpose())
    
    aux_ = []
    for j in range(model['dim']+1):
        aux_.append(convert_in[j].transform(inputs.transpose()[j].reshape(-1, 1)))
    input_scaled = np.stack(aux_).transpose()[0]
    
    # Predicting continuation values 
    # Scaling Neural Network outputs
    pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out.inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
    imm_pay = payoff(stock, model)
    
    # Array of realized payoffs
    # vals = np.where(np.logical_or(q_hat < imm_pay, imm_pay == 0), imm_pay, 0).astype(float)
    vals = np.repeat(0., nSims)
    # vals = copy.deepcopy(imm_pay)
    # List of arrays with stopping decisions
    # True means continue; False means stop 
    cont = [np.full(nSims, True)]
    # Initializing stopping times
    stop = np.repeat(float(model['T']), nSims)
    # stop = copy.deepcopy(times)
        
    stock = sim_gbm(stock, model)
    times += model['dt']
    t_min = min(times)
    
    # Forward Loop
    while t_min < model['T']:
        if display_time:
            start_time = time.time()
        
        # Scaling Neural Network inputs
        aux_ = [convert_in[0].transform(times.reshape(-1, 1))]
        for j in range(model['dim']):
            aux_.append(convert_in[j+1].transform(stock.transpose()[j].reshape(-1, 1)))
        input_scaled = np.stack(aux_).transpose()[0]
        
        # Predicting continuation values 
        # Scaling Neural Network outputs
        pred = NN.predict(input_scaled)
        prediction = np.ravel(convert_out.inverse_transform(pred))
        q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
        imm_pay = payoff(stock, model)
        
        # Updating the stopping decision
        logic = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), cont[-1])
        cont.append(logic)
        
        if (True in cont[-1]) == False:
            break
        # Perform stopping 
        for k in range(nSims):
            if times[k] == model['T']:
                if cont[-1][k]:
                    vals[k] = imm_pay[k]
                    stop[k] = times[k]
                    cont[-1][k] = False
            elif times[k] > model['T']:
                cont[-1][k] = False
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    vals[k] = imm_pay[k]
                    stop[k] = times[k]
                    
        stock = sim_gbm(stock, model)
        times += model['dt']
        t_min = min(times)
        
        # Displaying Time per Step
        if display_time:
            print('Time Min:', np.round(t_min-model['dt'], 4), 'Time:', \
                           np.round(time.time()-start_time, 2), 'sec')
    if (True in cont[-1]):
        imm_pay = payoff(stock, model)
        vals = np.where(cont[-1], imm_pay, vals).astype(float)    
    # Realized payoffs
    r_pay = np.exp(-model['r']*stop)*vals
    return (r_pay, stop)
        
