## Stage 3 Updating

# from stock_v4 import QMC_range
from stock_v5 import stock_sim_v5
# from stock_v4 import stock_sim_stop
from payoffs import payoff
from nn_timing import NN_timing
from nn_aggregate_v3 import NN_payoff_neo
import tensorflow as tf

## Libraries
import time
import numpy as np


def NN_loop_v17(loops, loop_size, model, NN, convert_in, convert_out, \
               nn_dt = None, stock_check = None, # ci = 0.99, 
               top = 0.2, epoch_num = 5, \
               batch_num = 64, opt_param = None, lossfct = None, display_time = False):
    '''
    Updates the aggreagte neural network 

    Parameters
    ----------
    loops : Integer with the number of self loops used in retraining the network
    loop_size : Integer with the number of simulations per loop
    model : Dictionary containing all the parameters of the stock and contract
    NN : Aggregate Neural Network object
    convert_in : List of input scaling objects (size model['dim']+1)
    convert_out : Output scaling object
    nn_dt : Float with the initial dt the aggregate network was trained on
    check : Number of loops at which the updated aggregate network is tested. 
            The default is None which corresponds to no testing.
    stock_check : Array of simulated stock paths used in testing the updated
                  aggregate network. The default is None which corresponds to no
                  testing.
    epoch_num : Number of epochs for retraining the aggregate network. The default is 50.
    batch_num : Number of batches for retraining the aggregate network. The default is 64.
    display_time : Display time spent per loop. The default is False.

    Returns
    -------
    NN : Updated Aggregate Neural Network object
    '''
    if nn_dt == None:
        raise ValueError('Provide the initial dt the aggregate network was trained on')
    if lossfct == None:
        lossfct = 'mean_squared_error'
    if opt_param == None:
        lr = 0.001
        b1 = 0.95
        b2 = 0.999
    else:
        lr = opt_param[0]
        b1 = opt_param[1]
        b2 = opt_param[2]
    
    nSteps = int(model['T']/model['dt'])        # Number of steps
    # n_pilot = nSteps*loop_size*model['dim']     # Number of pilot paths
    n_pilot = nSteps*loop_size                  # Number of pilot paths
    
    #low = (1-ci)/2*100
    #x_range = []
    sample_m = []
    explore = 10000
    sample_expl = []
    epsilon = 10**(-8)
    for t_step in np.round(np.arange(model['dt'], model['T']+epsilon, model['dt']),4):
        step = int(round(t_step/model['dt']))-1
        # Filtering and weighting based on the Lognormal distribution
        sample_expl.append([])
        for j in range(model['dim']):       # Lognormal Sampling
            sample_expl[step].append(np.random.lognormal(np.log(model['x0'][j]) + \
                    (model['r'] - model['div'][j] - model['sigma'][j]**2/2)*\
                    (t_step), model['sigma'][j]*np.sqrt(t_step), explore))
        sample_expl[step] = np.asarray(sample_expl[step], dtype=float).transpose()
        sample_expl[step] = np.delete(sample_expl[step], np.where(payoff(sample_expl[step], model) == 0), axis = 0)
        sample_m.append(len(sample_expl[step])/explore)
        #if step == nSteps-1:
        #    for j in range(model['dim']):
        #        x_range.append([np.percentile(sample_expl[step].transpose()[j], low),
        #                        np.percentile(sample_expl[step].transpose()[j], 100-low)])
    # x_range = np.asarray(x_range, dtype=float)
    sample_m = 1/np.asarray(sample_m, dtype=float)
    # x_qmc_all = QMC_range(explore, x_range, 'Halton', model)
    
    r_check = NN_payoff_neo(0, stock_check, model, 'agg', NN, convert_in, \
            convert_out, val = 'cont', nn_val = 'cont', nn_dt = nn_dt)
    max_check = np.mean(r_check)
    if display_time:
        print('Stage 2 Price:', np.round(max_check, 4))
    
    sample_x = []
    pref_sample = []
    rest_sample = []    
    for t_step in np.round(np.arange(model['dt'], model['T'], model['dt']),4):
        step = int(round(t_step/model['dt']))-1
        # Filtering and weighting based on the Lognormal distribution
        sample_x.append([])
        for j in range(model['dim']):       # Lognormal Sampling
            sample_x[step].append(np.random.lognormal(np.log(model['x0'][j]) + \
                    (model['r'] - model['div'][j] - model['sigma'][j]**2/2)*\
                    (t_step), model['sigma'][j]*np.sqrt(t_step), \
                    round(n_pilot*sample_m[step])))
        sample_x[step] = np.asarray(sample_x[step], dtype=float).transpose()
        sample_x[step] = np.delete(sample_x[step], np.where(payoff(sample_x[step], model) == 0), axis = 0)
        t_val_x = np.asarray(NN_timing(t_step, sample_x[step], NN, \
                    convert_in, convert_out, model, nn_dt=nn_dt), dtype=float)
        
        # Using the Gaussinan function to highlight samples close to the boundary
        gauss = 5*np.exp(-t_val_x**2/5)
        sort_arr = sorted(zip(gauss, range(len(t_val_x))))
        sort_el, idx = zip(*sort_arr)
        inputs_num = int(round(len(t_val_x)*top))    # top% of points near boundary
        pref_sample.append(np.asarray([sample_x[step][i] for i in idx[-inputs_num-1:-1]], dtype=float))
        rest_sample.append(np.asarray([sample_x[step][i] for i in idx[:-inputs_num-1]], dtype=float))
    
    # diff_r = []                 # List of option price differences across loops
    # lr_list = []                # List of learning rates across loops
    # time_val = []               # List of timing values at QMC points
    price_data = [max_check]    # List of option prices
    loop_max = None             # Loop at which the maximum option price is reached
    for loop in range(loops):
        if display_time:
            loop_time = time.time()
        
        if loop_max == loop - 1:
            # Update the sampling points when a new maximum is reached
            sample_x = []
            for t_step in np.round(np.arange(model['dt'], model['T'], model['dt']),4):
                step = int(round(t_step/model['dt']))-1
                
                # Sampling new points
                sample_x.append([])
                for j in range(model['dim']):       
                    sample_x[step].append(np.random.lognormal(np.log(model['x0'][j]) + \
                            (model['r'] - model['div'][j] - model['sigma'][j]**2/2)*\
                            t_step, model['sigma'][j]*np.sqrt(t_step), round(n_pilot*sample_m[step])))
                sample_x[step] = np.asarray(sample_x[step], dtype=float).transpose()
                sample_x[step] = np.delete(sample_x[step], np.where(payoff(sample_x[step], model) == 0), axis = 0)
                t_val_x = np.asarray(NN_timing(t_step, sample_x[step], NN, \
                            convert_in, convert_out, model, nn_dt=nn_dt), dtype=float)
                
                gauss = 5*np.exp(-t_val_x**2/5)
                sort_arr = sorted(zip(gauss, range(len(t_val_x))))
                sort_el, idx = zip(*sort_arr)
                inputs_num = int(round(len(t_val_x)*top))
                
                pref_sample_new = np.asarray([sample_x[step][i] for i in idx[-inputs_num-1:-1]], dtype=float)
                ### plt.scatter(pref_sample_new.transpose()[0], pref_sample_new.transpose()[1], s = 0.15)
                rest_sample_new = np.asarray([sample_x[step][i] for i in idx[:-inputs_num-1]], dtype=float)
                ### plt.scatter(rest_sample_new.transpose()[0], rest_sample_new.transpose()[1], s = 0.15)
                pref_merge = np.concatenate((pref_sample[step], pref_sample_new))
                rest_merge = np.concatenate((rest_sample[step], rest_sample_new))
                
                # Selecting the points
                idx_p = np.random.choice(range(len(pref_merge)), \
                            size=int(round(len(pref_merge)/2)), replace=False)
                pref_sample[step] = np.asarray([pref_merge[i] for i in idx_p], dtype=float)
                idx_r = np.random.choice(range(len(rest_merge)), \
                            size=int(round(len(rest_merge)/2)), replace=False)
                rest_sample[step] = np.asarray([rest_merge[i] for i in idx_r], dtype=float)
        
        # Selecting In-the-Money paths for the Aggregate Neural Network training
        x_agg = [] 
        y_agg = []
        for t_step in np.round(np.arange(model['dt'], model['T'], model['dt']),4):
            step = int(round(t_step/model['dt']))-1
            # Selecting sample points for training
            idx1 = np.random.choice(range(len(pref_sample[step])), \
                        size=int(round(loop_size/2)), replace=False)
            arr1 = np.asarray([pref_sample[step][i] for i in idx1], dtype=float)
            ### plt.scatter(arr1.transpose()[0], arr1.transpose()[1], s = 0.15)
            idx2 = np.random.choice(range(len(rest_sample[step])), \
                        size=int(round(loop_size/2)), replace=False)
            arr2 = np.asarray([rest_sample[step][i] for i in idx2], dtype=float)
            ### plt.scatter(arr2.transpose()[0], arr2.transpose()[1], s = 0.15)
            
            s0 = np.concatenate((arr1, arr2))
            stock = stock_sim_v5(t_step, s0, model)
            
            # Finding continuation values when stopping at the current
            # time step is not allowed
            q_val = NN_loop_pay_v8(t_step, stock, model, NN, convert_in, convert_out, nn_dt)
            # x = np.delete(stock, np.where(payoff(stock[0], model) == 0), axis = 1)[0]
            # y = np.delete(q_val, np.where(payoff(stock[0], model) == 0), axis = 0)
            x = s0
            y = q_val
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
        
        # Save the learning rate
        # lr_list.append(lr)
        # Defining and training the aggregate neural network
        opt = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = b1, beta_2 = b2)
        NN.compile(optimizer = opt, loss = lossfct)
        NN.fit(x_, y_, epochs = epoch_num, batch_size = batch_num, verbose = 0)
        
        # Computing the option price for the validation paths
        r_check = NN_payoff_neo(0, stock_check, model, 'agg', NN, convert_in, \
                        convert_out, val = 'cont', nn_val = 'cont', nn_dt = nn_dt)
        
        r_curr = np.mean(r_check)
        # diff_r.append(r_curr - price_data[-1])
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
            # print('Loop:',loop+1,'New Max')
            loop_max = loop
        
        # Computing QMC timing values
        # time_val.append(NN_timing(0, x_qmc_all, NN, convert_in, convert_out, model, nn_dt=nn_dt))
        
        # Stop when the learning rate gets very small
        if lr < 5*10**(-8):
            break
        
        # Displaying Time per Loop
        if display_time:
            print('Loop:', loop+1,'LR:', np.round(lr,9),'Price:', np.round(r_curr, 4),\
                  'Paths', len(y_), 'Time:', np.round(time.time()-loop_time,2), 'sec')
    # return (x_qmc_all, time_val, lr_list, diff_r, NN_max)     
    return NN_max

def NN_loop_pay_v8(t_step, stock, model, NN, convert_in, convert_out, nn_dt,\
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
    stopT : Array of stopping times (size N)
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
    
    # Initializing stopping times
    if stop:
        stopT = np.repeat(model['T'], nSims) 
    
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
        q_hat = np.exp(-model['r']*nn_dt)*q_hat
        imm_pay = payoff(stock[i-step], model)
        
        # Updating the stopping decision
        if i == step:
            # No stopping at the current time step
            logic = np.full(nSims, True)
        else:
            logic = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), cont[-1])
        cont.append(logic)
        
        # Perform stopping 
        v_cont = np.exp(model['r']*model['dt'])*v_cont
        for k in range(nSims):
            if i == step:
                if cont[-1][k] == False:
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['dt']
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_cont[k] = imm_pay[k]
                    # Updating stopping times
                    if stop:
                        stopT[k] = (i+1)*model['dt']
        
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
        return (r_pay, stopT)
    else:
        return r_pay
        