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
Use this file to run code for Stage 1 and Stage 2 testing (see description of each in READ.me).

'''


'''
Stage 1 testing
---------------

To test various option contract parameters, change the input dictionary (bput1 in line 38). 
Models from mlOSP are given in model_parameters.py; any of those variables can be imported to 
this file to replace the current input dictionary. 

To test various neural network hyperparameters, change manually (epnum, node_num, etc) 
in this file and also in NN_seq_train_neo from nn_train_neo.py

'''

#########################
#### Stage 1 Testing ####
#########################

# 1D Basket Put
bput1 = {'dim': 1, 'K': 40, 'x0': np.repeat(40, 1), 'sigma': np.repeat(0.2,1), 
          'r': 0.06, 'div': np.repeat(0, 1), 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'}

# Initializing -- Fine Time Grid
model_update(mcall2, dt = 1/3)
np.random.seed(15)
tf.random.set_seed(15)
stock_fine_m2 = stock_sim(100000, mcall2)
(c_in_m2, c_out_m2) = scaler(stock_fine_m2, mcall2)

# Thin at coarse dt
coa_dt = 0.04
model_update(bput1, dt = coa_dt)
stock_coa_p1 = stock_thin(stock_fine_p1, bput1, coa_dt) # Coarse time grid
  
# Training Data -- Coarse Time Grid
epnum = 50
np.random.seed(15)
tf.random.set_seed(15)
(NN_seq_coa_p1, x_p1, y_p1) = NN_seq_train_neo(stock_coa_p1, bput1, c_in_p1, c_out_p1, \
                                    theta = 'average', data = True, val = 'cont', \
                                    node_num = 60, epoch_num = epnum, display_time=True, tot_time = True)

# ## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_p1 = stock_sim(100000, bput1)
coa_dt = 0.04
model_update(bput1, dt = coa_dt)
stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, coa_dt)

(r_seq_coa_p1, stop_seq_p1) = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'seq', NN_seq_coa_p1, c_in_p1, c_out_p1, \
                          val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)

print('dt:', coa_dt)
print('Price - seq NN:', np.round(np.mean(r_seq_coa_p1), 4))

## Boundary Plot
# Sequence of Neural Networks 
(x,y,z) = NN_bound_neo(NN_seq_coa_p1, c_in_p1, c_out_p1, bput1, net = 'seq', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
            origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.colorbar()
#plt.plot(np.arange(0, bput1['dt'] + bput1['T'], bput1['dt']), np.repeat(bput1['K'], 21), linewidth=1, color= 'black', linestyle='dashdot')
plt.xlabel('Time')
plt.ylabel('Stock Price')
#plt.title('1-D Put Boundary Plot')
# plt.title('!D Put Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(np.mean(r_seq_coa_p1),4)))
plt.savefig('Basket-Put-Boundary-seq-NN-dt-'+str(coa_dt)+'.png', dpi=1000)
plt.clf()



'''
Stage 2 testing
---------------

To test various option contract parameters, change the input dictionary (bput1 in line 38). 
Models from mlOSP are given in model_parameters.py; any of those variables can be imported to 
this file to replace the current input dictionary. 

To test various neural network hyperparameters, change manually (epnum, node_num, etc) 
in this file and also in NN_saggregate_neo from nn_aggregate_v3.py


'''

### Stage 2 


mcall2 = {'dim': 5, 'K': 100, 'x0': np.repeat(70, 5), 'sigma': [0.08,0.16,0.24,0.32,0.4], 
          'r': 0.05, 'div': np.repeat(0.1,5), 'T': 3, 'dt': (1/3), 'payoff.func': 'maxi.call.payoff'} # cosmin: mcall1

   
## Aggregate Neural Network -- Coarse Time Grid
np.random.seed(16)
tf.random.set_seed(16)
# Same Paths
NN_agg_coa_p1 = NN_aggregate_neo(bput1, NN_seq_coa_p1, c_in_p1, c_out_p1, nn_val = 'cont', \
                            stock = stock_coa_p1, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
# Training Data
np.random.seed(16)
tf.random.set_seed(16)
NN_agg_coa_data_p1 = NN_aggregate_neo(bput1, NN_seq_coa_p1, c_in_p1, c_out_p1, nn_val = 'cont', \
                            data = True, x = x_p1, y = y_p1, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_p1 = stock_sim(100000, bput1)
coa_dt = 0.2

for dt in [0.2, 0.1, 0.05, 0.025]:
    start_time = time.time()
    model_update(bput1, dt = dt)
    stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, dt) # Coarse Test Grid
    # Same Paths
    (r_agg_coa_p1, stop_agg_p1) = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_p1, c_in_p1, c_out_p1, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
        
    # Training Data 
    (r_agg_coa_data_p1, stop_agg_data_p1) = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_data_p1, c_in_p1, c_out_p1, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
     
    print('dt:', dt)
    # Same Paths  
    print('Price - agg NN - Path:', np.round(np.mean(r_agg_coa_p1), 4))
    print('Avg Std agg NN - Path:', np.round(np.std(r_agg_coa_p1, ddof = 1)/np.sqrt(len(stockFwd_coa_p1[0])), 4))
    # Training Data 
    print('Price - agg NN - Data:', np.round(np.mean(r_agg_coa_data_p1), 4))
    print('Avg Std agg NN - Data:', np.round(np.std(r_agg_coa_data_p1, ddof = 1)/np.sqrt(len(stockFwd_coa_p1[0])), 4))
    
    # ## Realized Payoff Comparisons
    # # Data vs. Same Paths
    # plt.scatter(r_agg_coa_data_p1[:5000], r_agg_coa_p1[:5000], s=1, c='black', marker='o')
    # plt.plot([-0.2, 10], [-0.2, 10], color = 'red', linewidth=0.5)
    # plt.xlabel('Aggregate - Data')
    # plt.ylabel('Aggregate - Same Paths')
    # plt.title('Realized Payoff - dt: '+str(dt))
    # plt.savefig('Aggregate-vs-Aggregate-1D-Put-train-'+str(coa_dt)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)
    # plt.clf()
