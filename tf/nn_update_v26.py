## Stage 3 Updating

from model_update import model_update
from stock_v6 import stock_sim
from stock_v6 import stock_sim_trunc
from payoffs import payoff
from nn_aggregate_v4 import NN_payoff_mt


## Libraries
import time
import numpy as np
import tensorflow as tf

def NN_loop_v26(loops, loop_size, model, NN, convert_in, convert_out, target, \
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
    aux_dt = model['dt']
    model_update(model, dt = target)
    
    if lossfct == None:
        lossfct = 'mean_squared_error'
    if opt_param == None:
        lr, b1, b2 = 0.001, 0.95, 0.999
    else:
        lr, b1, b2 = opt_param[0], opt_param[1], opt_param[2]
    if type(stock_check) != type(np.array([])):
        stock_check = stock_sim(loop_size, model)
    
    nSteps = int(model['T']/model['dt'])        # Number of steps
    n_pilot = min(nSteps, 5)*loop_size          # Number of pilot paths
    
    
    # Computing a multiplier
    # Multiplier = 1/(fraction of paths in-the-money)
    multiplier = 0
    stock_m = stock_sim(loop_size, model)
    (r_m, stop_m) = NN_payoff_mt(0, stock_m, model, 'agg', NN, convert_in, \
            convert_out, stop = True)
    for k in range(loop_size):
        if stop_m[k] == model['T']:
            multiplier += 1
    multiplier = 1/(1-multiplier/loop_size)
    
    # Building large samples of positive and negative timing values
    stock_sample = stock_sim(round(multiplier*n_pilot), model)
    r_val, stop_sample = NN_payoff_mt(0, stock_sample, model, 'agg', NN, convert_in, \
            convert_out, stop = True)
    sample_t_neg = []
    sample_t_pos = []
    for k in range(len(stop_sample)):
        if stop_sample[k] != model['T']:
            step = int(round(stop_sample[k]/model['dt']))-1
            sample_t_neg.append([stock_sample[step][k], stop_sample[k]])
            if step-1 >= 0:
                sample_t_pos.append([stock_sample[step-1][k], stop_sample[k]-model['dt']])    
    sample_t_neg = np.asarray(sample_t_neg, dtype=object)
    sample_t_pos = np.asarray(sample_t_pos, dtype=object)
    
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
        
        # Update sample of positive and negative timing values
        if loop_max == loop - 1 or count_update >= 5:
            count_update = 0
            # Update stopping
            stock_sample = stock_sim(round(multiplier*n_pilot), model)
            (r_val, stop_sample) = NN_payoff_mt(0, stock_sample, model, 'agg', NN, \
                                                convert_in, convert_out, stop = True)
            sample_t_neg = []
            sample_t_pos = []
            for k in range(len(stop_sample)):
                if stop_sample[k] != model['T']:
                    step = int(round(stop_sample[k]/model['dt']))-1
                    sample_t_neg.append([stock_sample[step][k], stop_sample[k]])
                    if step-1 >= 0:
                        sample_t_pos.append([stock_sample[step-1][k], stop_sample[k]-model['dt']])    
            sample_t_neg = np.asarray(sample_t_neg, dtype=object)
            sample_t_pos = np.asarray(sample_t_pos, dtype=object)
        else:
            count_update += 1
            
        # Selecting In-the-Money paths for the Aggregate Neural Network training
        x_agg = [] 
        y_agg = []        
        
        # Extracting subsamples from the samples of positive and negative timing values
        # Ratio of 10% to 90%
        idx0 = np.random.choice(range(loop_size), \
                    size=int(round(loop_size*0.05)), replace=False)
        arr0_time = np.random.choice(range(nSteps-1), \
                    size=int(round(loop_size*0.05)), replace=True)
        arr0_loc = np.asarray([stock_m[arr0_time[k]][idx0[k]] for k in range(len(idx0))], dtype=float)
        arr0_time = np.round((arr0_time+1)*model['dt'], 4)
        ### plt.scatter(arr0_loc.transpose()[0], arr0_loc.transpose()[1], s = 0.15)
        ### plt.savefig('Other-'+str(loop+1)+'.png', dpi=1000)
        ### plt.clf()
        
        # Ratio of positive to negative timing values is 1:1
        idx1 = np.random.choice(range(len(sample_t_neg)), \
                    size=int(round(loop_size*0.95/2)), replace=False)
        arr1_loc = np.asarray([sample_t_neg[i][0] for i in idx1], dtype=float)
        arr1_time = np.asarray([sample_t_neg[i][1] for i in idx1], dtype=float)
        arr1_time = np.round(arr1_time, 4)
        ### plt.scatter(arr1_loc.transpose()[0], arr1_loc.transpose()[1], s = 0.15)
        ### plt.savefig('Neg-TV-'+str(loop+1)+'.png', dpi=1000)
        ### plt.clf()
        
        idx2 = np.random.choice(range(len(sample_t_pos)), \
                    size=int(round(loop_size*0.95/2)), replace=False)
        arr2_loc = np.asarray([sample_t_pos[i][0] for i in idx2], dtype=float)
        arr2_time = np.asarray([sample_t_pos[i][1] for i in idx2], dtype=float)
        arr2_time = np.round(arr2_time, 4)
        ### plt.scatter(arr2_loc.transpose()[0], arr2_loc.transpose()[1], s = 0.15)
        ### plt.savefig('Pos-TV-'+str(loop+1)+'.png', dpi=1000)
        ### plt.clf()
        
        # Merging subsamples
        s0_loc = np.concatenate((arr0_loc, arr1_loc, arr2_loc))
        s0_time = np.concatenate((arr0_time, arr1_time, arr2_time))
        
        # Formatting the data into a 2D array
        s0 = [[] for i in range(nSteps-1)]
        for k in range(loop_size):
            step = int(round(s0_time[k]/model['dt']))-1
            s0[step].append(list(s0_loc[k])) 
            
        for t_step in np.round(np.arange(model['dt'], model['T'], model['dt']), 4):
            step = int(round(t_step/model['dt']))-1
            s0[step] = np.asarray(s0[step], dtype = float)
            if len(s0[step]) == 0:
                continue
            stock = stock_sim_trunc(t_step, s0[step], model)
            
            # Finding continuation values when stopping at the current
            # time step is not allowed
            x = s0[step]
            y = NN_loop_pay_v10(t_step, stock, model, NN, convert_in, convert_out)
            
            x_agg.append(np.hstack((np.repeat(t_step, len(y)).reshape(len(y), 1), x)))
            y_agg.append(y)
            
        # Training data for the aggregate network
        x = np.concatenate(x_agg, axis = 0)
        y = np.concatenate(y_agg, axis = 0)
        
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
        
        # go along paths and t value
            
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
        
        # Displaying Time per Loop
        if display_time:
            print('Loop:', loop+1,'LR:', np.round(lr,9),'Price:', np.round(r_curr, 4),\
                  'Paths', len(y_), 'Time:', np.round(time.time()-loop_time,2), 'sec')    
    model_update(model, dt = aux_dt)
    return NN_max

def NN_loop_pay_v10(t_step, stock, model, NN, convert_in, convert_out, \
                   stop = False, display_time = False):
    '''
    Longstaff Schwartz Algorithm
    Computes the loop payoff at a given time step of an aggregate network.

    Parameters
    ----------
    t_step : Float indicating the initial time step
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    NN : Neural Network object                        
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling objects
    stop : Boolean asserting whether to display stopping times. The default is False.
    display_time : Boolean asserting whether to display the time spent per step. 
                   The default is False.

    Returns
    -------
    r_pay : Array of realized payoff (size N)
    stopT : Array of stopping times (size N) - not allowed to stop at t_step
    stopN : Array of stopping times (size N) - may stop at t_step
    '''
    
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    
    # Transforming the time step into an integer
    step = int(np.round(t_step/model['dt']))-1
    if step >= nSteps-1:
        raise ValueError('Time step is equal to or exceeds maturity.')
    
    # Array of continuation values
    v_cont = payoff(stock[0], model)
    # List of arrays with stopping decisions
    # True means continue; False means stop 
    cont = []
    if stop:
        contN = []
    
    # Initializing stopping times
    if stop:
        stopT = np.repeat(float(model['T']), nSims) # No stopping at current time step
        stopN = np.repeat(float(model['T']), nSims) # May stop at current time step
    
    # Forward Loop
    for i in range(step, nSteps-1):
        if display_time:
            start_time = time.time()
        
        # Scaling Neural Network inputs
        aux_ = [convert_in[0].transform(np.repeat((i+1)/nSteps*model['T'], nSims).reshape(-1, 1))]
        for j in range(model['dim']):
            aux_.append(convert_in[j+1].transform(stock[i-step].transpose()[j].reshape(-1, 1)))
        input_scaled = np.stack(aux_).transpose()[0]
        
        # Predicting continuation values 
        # Scaling Neural Network outputs
        pred = NN.predict(input_scaled)
        prediction = np.ravel(convert_out.inverse_transform(pred))
        q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
        imm_pay = payoff(stock[i-step], model)
        
        # Updating the stopping decision
        if i == step:
            # No stopping at the current time step
            logic = np.full(nSims, True)
            if stop:
                logicN = np.logical_or(q_hat > imm_pay, imm_pay == 0)
        else:
            logic = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), cont[-1])
            if stop:
                logicN = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), contN[-1])
        cont.append(logic)
        if stop:
            contN.append(logicN)
        
        # Perform stopping 
        v_cont = np.exp(model['r']*model['dt'])*v_cont
        for k in range(nSims):
            if i == step:
                if cont[-1][k] == False:
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['dt']
                if stop:
                    if contN[-1][k] == False:
                        stopN[k] = (i+1)*model['dt']
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['dt']
                if stop:
                    if (contN[-1][k] == False) and (contN[-2][k] == True):
                        stopN[k] = (i+1)*model['dt']
                    
        
        # Displaying Time per Step
        if display_time:
            print('Step:', np.round((i+1)*model['dt'], 2), 'Time:', \
                           np.round(time.time()-start_time, 2), 'sec')
    
    # Computing the terminal payoff
    imm_pay = payoff(stock[-1], model)
    v_cont = np.exp(model['r']*model['dt'])*v_cont
    for k in range(nSims):
        if cont[-1][k] == True:
            v_cont[k] = imm_pay[k]
    
    # Realized payoffs
    r_pay = np.exp(-model['r']*model['dt']*(nSteps-1-step))*v_cont
    if stop:
        return (r_pay, stopT, stopN)
    else:
        return r_pay
        
