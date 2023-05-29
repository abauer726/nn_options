# Stage 3 - Implementation

from scaler import scaler
from stock_v4 import stock_sim
from stock_v4 import stock_thin
from nn_train_neo import NN_seq_train_neo
from nn_aggregate_v3 import NN_payoff_neo
from nn_aggregate_v3 import NN_aggregate_neo
from plots_neo import NN_contour_neo
from model_update import model_update
# from nn_timing import NN_timing
from nn_update_v17 import NN_loop_v17

## Libraries
import time
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf


'''
This file includes code used to run Stage 3 (see READ.me for description). 
Stage 1 and 2 code is also included for reference. 

To test various option contract parameters, change the input dictionary (mcall2 in line 38). 
Models from mlOSP are given in model_parameters.py; any of those variables can be imported to 
this file to replace the current input dictionary. 

To test various neural network hyperparameters, change manually (epnum, node_num, etc) 
in this file and also in the function files (NN_loop_v17 from nn_update_v17 to change # of loops)

'''



mcall2 = {'dim': 2, 'K': 100, 'x0': np.repeat(100, 2), 'sigma': np.repeat(0.2,2), 
         'r': 0.05, 'div': np.repeat(0.1,2), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}


#######################
###### Stage One ######
#######################


# Initializing -- Fine Time Grid
model_update(mcall2, dt = 1/3)
np.random.seed(15)
tf.random.set_seed(15)
stock_fine_m2 = stock_sim(100000, mcall2)
(c_in_m2, c_out_m2) = scaler(stock_fine_m2, mcall2)

# Thin at coarse dt
coa_dt = 1/3
model_update(mcall2, dt = coa_dt)
stock_coa_m2 = stock_thin(stock_fine_m2, mcall2, coa_dt) # Coarse time grid

## Sequence of Neural Networks -- Coarse Time Grid
epnum = 5
lr = 0.001
b1 = 0.95
b2 = 0.999
opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
np.random.seed(15)
tf.random.set_seed(15)
(NN_seq_coa_m2, x_m2, y_m2) = NN_seq_train_neo(stock_coa_m2, mcall2, c_in_m2, c_out_m2, \
                                theta = 'average', data = True, val = 'cont', \
                                node_num = 25, epoch_num = epnum, optim = opt, display_time=False)

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(mcall2, dt = 1/3)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_m2 = stock_sim(100000, mcall2)
model_update(mcall2, dt = coa_dt)
stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, coa_dt)

# Computing the option price
r_seq_m2 = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'seq', NN_seq_coa_m2, c_in_m2, \
                              c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
print('Stage 1 Price:', np.round(np.mean(r_seq_m2), 4))

# Contours of the Sequence of Neural Networks
model_update(mcall2, dt = 1/3)
norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
contours = []

# t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
for t in np.arange(mcall2['dt'], mcall2['T'], mcall2['dt']):
    start_time = time.time()
    ## Contour
    (x,y,z) = NN_contour_neo(t, NN_seq_coa_m2, c_in_m2, c_out_m2, mcall2, net = 'seq', \
                             down = 0.8, up = 1.8, display_time = False)
    
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[-1], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
               origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
    plt.colorbar()
    plt.plot(np.array([80,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.plot(np.array([100,100]), np.array([80,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.title('Stage 1 - Map: '+str(np.round(t, 4))+' Price: '+str(np.round(np.mean(r_seq_m2),4)))
    plt.savefig('MC-Stage-1-ReLU-Map-'+str(np.round(t, 4))+'.png', dpi=1000)
    plt.clf()
    print('Map:', np.round(t,3),'Time:', np.round(time.time()-start_time,2), 'sec')

### STAGE 2 ###

# Shuffling data
np.random.seed(0)
merge = np.append(x_m2.transpose(), y_m2.transpose()).reshape(mcall2['dim']+2, len(y_m2)).transpose()
np.random.shuffle(merge)
x_r_m2 = merge.transpose()[:-1].transpose()
y_r_m2 = merge.transpose()[-1].reshape(-1, 1) 

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(mcall2, dt = 1/3)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_m2 = stock_sim(100000, mcall2)
model_update(mcall2, dt = coa_dt)
stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, coa_dt)

# Stage 2: Initializing the Aggregate Network
epnum = 5
lr = 0.001
b1 = 0.95
b2 = 0.999
opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
np.random.seed(16)
tf.random.set_seed(16)
NN_agg_m2 = NN_aggregate_neo(mcall2, NN_seq_coa_m2, c_in_m2, c_out_m2, \
                        nn_val = 'cont', data = True, x = x_r_m2, y = y_r_m2, \
                        node_num = 25, batch_num = 64, epoch_num = epnum, \
                        optim = opt, display_time=False)

# Computing the option price
(r_agg_stage2_m2, stop_stage2_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_m2, c_in_m2, \
                              c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, 
                              stop = True, display_time=False)
        
print('Stage 2 Price:', np.round(np.mean(r_agg_stage2_m2), 4))

# Contours 80 - 180
# t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
for t in np.arange(mcall2['dt'], mcall2['T'], mcall2['dt']):
    start_time = time.time()
    ## Contour
    (x,y,z) = NN_contour_neo(t, NN_agg_m2, c_in_m2, c_out_m2, mcall2, net = 'agg', \
                             nn_val = 'cont', nn_dt = coa_dt, down = 0.8, up = 1.8, display_time = False)
    
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[-1], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
               origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
    plt.colorbar()
    plt.plot(np.array([80,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.plot(np.array([100,100]), np.array([80,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.title('Stage 2 - Map: '+str(np.round(t, 2))+' Price: '+str(np.round(np.mean(r_agg_stage2_m2),4)))
    plt.savefig('MC-Stage-2-ReLU-Map-'+str(np.round(t, 2))+'-wide.png', dpi=1000)
    plt.clf()
    print('Stage 2 Map:', np.round(t,3), 'Price:', np.round(np.mean(r_agg_stage2_m2), 4), \
          'Time:', np.round(time.time()-start_time,2), 'sec')

### STAGE 3 ###

### Setting up the validating paths
# model_update(mcall2, dt = 1/3)
np.random.seed(99)
tf.random.set_seed(99)
stock_valid_m2 = stock_sim(10000, mcall2)

np.random.seed(16)
tf.random.set_seed(16)
loops = 40
loop_size = 500
epnum_s3 = 5
# For specific learning rate
# optp = [0.001, 0.95, 0.999]

# Looping
(x_qmc_m2, time_val_m2, lr_m2, dif_m2, NN_max_m2) = NN_loop_v17(loops, loop_size, mcall2, \
        NN_agg_m2, c_in_m2, c_out_m2, nn_dt = coa_dt, \
        stock_check = stock_valid_m2, ci = 0.99, epoch_num = epnum_s3, display_time = True)

NN_max_m2 = NN_loop_v17(loops, loop_size, mcall2, NN_agg_m2, c_in_m2, c_out_m2, \
                        nn_dt = coa_dt, stock_check = stock_valid_m2, epoch_num = epnum_s3, \
                        display_time = True)
    
## Testing 
# Final Loop Price
(r_agg_stage3_m2, stop_stage3_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', \
            NN_agg_m2, c_in_m2, c_out_m2, val = 'cont', nn_val = 'cont', \
            nn_dt = coa_dt, stop = True, display_time=False)
# Max Loop Price
(r_max_stage3_m2, stop_max_stage3_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', \
            NN_max_m2, c_in_m2, c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, \
            stop = True, display_time=False)
    
print('Stage 3 -Final- Price:', np.round(np.mean(r_agg_stage3_m2), 4))
print('Stage 3 -Max- Price:', np.round(np.mean(r_max_stage3_m2), 4))
