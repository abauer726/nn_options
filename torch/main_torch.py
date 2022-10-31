#### EDIT THIS TO BE FOR PYTORCH 
# as of 10/17: this page is entirely in tensorflow


# Stage 1 and 2 - Diagnostics

from scaler import scaler
from stock_v5 import stock_sim
from stock_v5 import stock_thin
from nn_train_neo_torch import NN_seq_train_neo
from nn_aggregate_v3_torch import NN_payoff_neo
from nn_aggregate_v3_torch import NN_aggregate_neo
from plots_neo import NN_bound_neo
from plots_neo import NN_contour_neo
from plots_neo import NN_time_neo
from model_update import model_update
from payoffs import payoff

## Libraries
import time
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf # delete this line when converted to torch
import torch


#########################
#### 1D - Basket Put ####
#########################

bput1 = {'dim': 1, 'K': 40, 'x0': np.repeat(40, 1), 'sigma': np.repeat(0.2,1), 
         'r': 0.05, 'div': np.repeat(0, 1), 'T': 1, 'dt': 0.05, 'payoff.func': 'put.payoff'}

# Initializing -- Fine Time Grid
model_update(bput1, dt = 0.025)
np.random.seed(15)
torch.manual_seed(15)
stock_fine_p1 = stock_sim(100000, bput1)
(c_in_p1, c_out_p1) = scaler(stock_fine_p1, bput1)

# Thin at coarse dt
coa_dt = 0.05
model_update(bput1, dt = coa_dt)
stock_coa_p1 = stock_thin(stock_fine_p1, bput1, coa_dt) # Coarse time grid
  
# Training Data -- Coarse Time Grid
epnum = 50
np.random.seed(15)
torch.manual_seed(15)
(NN_seq_coa_p1, x_p1, y_p1) = NN_seq_train_neo(stock_coa_p1, bput1, c_in_p1, c_out_p1, \
                                    theta = 'average', data = True, val = 'cont', \
                                    node_num = 25, epoch_num = epnum, display_time=True)

## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
torch.manual_seed(15)
stockFwd_fine_p1 = stock_sim(100000, bput1)
coa_dt = 0.05
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
plt.plot(np.arange(0, bput1['dt'] + bput1['T'], bput1['dt']), np.repeat(bput1['K'], 21), linewidth=1, color= 'black', linestyle='dashdot')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('1-D Put Boundary Plot')
# plt.title('!D Put Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(np.mean(r_seq_coa_p1),4)))
plt.savefig('Basket-Put-Boundary-seq-NN-dt-'+str(coa_dt)+'.png', dpi=1000)
plt.clf()

### Stage 2 

'''    
## Aggregate Neural Network -- Coarse Time Grid
np.random.seed(16)
torch.manual_seed(16)
# Same Paths
NN_agg_coa_p1 = NN_aggregate_neo(bput1, NN_seq_coa_p1, c_in_p1, c_out_p1, nn_val = 'cont', \
                            stock = stock_coa_p1, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
# Training Data
np.random.seed(16)
torch.manual_seed(16)
NN_agg_coa_data_p1 = NN_aggregate_neo(bput1, NN_seq_coa_p1, c_in_p1, c_out_p1, nn_val = 'cont', \
                            data = True, x = x_p1, y = y_p1, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
torch.manual_seed(16)
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
    
    ## Realized Payoff Comparisons
    # Data vs. Same Paths
    plt.scatter(r_agg_coa_data_p1[:5000], r_agg_coa_p1[:5000], s=1, c='black', marker='o')
    plt.plot([-0.2, 10], [-0.2, 10], color = 'red', linewidth=0.5)
    plt.xlabel('Aggregate - Data')
    plt.ylabel('Aggregate - Same Paths')
    plt.title('Realized Payoff - dt: '+str(dt))
    plt.savefig('Aggregate-vs-Aggregate-1D-Put-train-'+str(coa_dt)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)
    plt.clf()

    # Aggregate Neural Network 
    # Same Paths 
    (x,y,z) = NN_bound_neo(NN_agg_coa_p1, c_in_p1, c_out_p1, bput1, net = 'agg', \
                           nn_val = 'cont', nn_dt = coa_dt, display_time = False)

    norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    contour = plt.contour(x, y, z, [0], colors='black')
    plt.clabel(contour, inline=True, fontsize=5)
    plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
               origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Boundary Plot - dt: '+str(dt)+' Price: '+str(np.round(np.mean(r_agg_coa_p1),4)))
    plt.savefig('Basket-Put-Boundary-agg-NN-same-path-train-'+str(coa_dt)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)
    plt.clf()

    # Training Data 
    (x,y,z) = NN_bound_neo(NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, net = 'agg', \
                           nn_val = 'cont', nn_dt = coa_dt, display_time = False)

    norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    contour = plt.contour(x, y, z, [0], colors='black')
    plt.clabel(contour, inline=True, fontsize=5)
    plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
               origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Boundary Plot - dt: '+str(dt)+' Price: '+str(np.round(np.mean(r_agg_coa_data_p1),4)))
    plt.savefig('Basket-Put-Boundary-agg-NN-data-train-'+str(coa_dt)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)
    plt.clf()
    print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,2), 'sec')

###########################
#### FORENSICS & PLOTS ####
###########################

## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_p1 = stock_sim(100000, bput1)
coa_dt = 0.2
model_update(bput1, dt = coa_dt)
stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, coa_dt) 
r_seq_coa_p1 = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'seq', NN_seq_coa_p1, c_in_p1, c_out_p1, \
                         val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=False, display_time=False)

## Boundary Plot
# Sequence of Neural Networks 
(x,y,z) = NN_bound_neo(NN_seq_coa_p1, c_in_p1, c_out_p1, bput1, net = 'seq', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(np.mean(r_seq_coa_p1),4)))
plt.savefig('Basket-Put-Boundary-seq-NN-dt-'+str(coa_dt)+'-forensics-time.png', dpi=1000)
plt.clf()

# Aggregate Neural Network 
# Same Paths 
(x,y,z) = NN_bound_neo(NN_agg_coa_p1, c_in_p1, c_out_p1, bput1, net = 'agg', nn_val = 'cont', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(np.mean(r_agg_coa_p1),4)))
plt.savefig('Basket-Put-Boundary-agg-NN-same-path-dt-'+str(coa_dt)+'-forensics-time.png', dpi=1000)
plt.clf()

# Training Data 
(x,y,z) = NN_bound_neo(NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, net = 'agg', nn_val = 'cont', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(np.mean(r_agg_coa_data_p1),4)))
plt.savefig('Basket-Put-Boundary-agg-NN-data-dt-'+str(coa_dt)+'-forensics-time.png', dpi=1000)
plt.clf()


# 38
pos = 35
t_inc = 0.01
# Same Paths
(x_path, y_path) = NN_time_neo(np.array([pos]), NN_agg_coa_p1, c_in_p1, c_out_p1, bput1, \
                       nn_val = 'cont', nn_dt = coa_dt, t_inc=t_inc, display_time = False)    
# Training Data
(x_data, y_data) = NN_time_neo(np.array([pos]), NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, \
                       nn_val = 'cont', nn_dt = coa_dt, t_inc=t_inc, display_time = False)

plt.plot(x_path, y_path, linewidth=1)
plt.plot(x_data, y_data, linewidth=1)
plt.legend(['Path','Data'])
plt.xlabel('Time')
plt.ylabel('Timing Value')
plt.title('Timing Values dt: '+str(t_inc)+' Position: '+str(pos))
plt.savefig('Timing-values-dt-'+str(dt)+'-pos-'+str(pos)+'-forensics-time.png', dpi=1000)
plt.clf()

## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_p1 = stock_sim(100000, bput1)
# Coarse Test Grid
coa_dt = 0.2
model_update(bput1, dt = coa_dt)
stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, coa_dt) 

# Select Path
path = 8

for dt in [0.2, 0.1, 0.05, 0.025]:
    start_time = time.time()
    model_update(bput1, dt = dt)
    stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, dt) # Coarse Test Grid
    
    # Same Paths
    (r_agg_p1, stop_agg_p1) = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_p1, c_in_p1, c_out_p1, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
        
    # Training Data 
    (r_agg_data_p1, stop_agg_data_p1) = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_data_p1, c_in_p1, c_out_p1, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
    
    print('dt:', dt)
    # Same Paths  
    print('Price - agg NN - Path:', np.round(np.mean(r_agg_p1), 4))
    print('Avg Std agg NN - Path:', np.round(np.std(r_agg_p1, ddof = 1)/np.sqrt(len(stockFwd_coa_p1[0])), 4))
    # Training Data 
    print('Price - agg NN - Data:', np.round(np.mean(r_agg_data_p1), 4))
    print('Avg Std agg NN - Data:', np.round(np.std(r_agg_data_p1, ddof = 1)/np.sqrt(len(stockFwd_coa_p1[0])), 4))
    
    # Aggregate Neural Network 
    # Same Paths 
    (x,y,z) = NN_bound_neo(NN_agg_coa_p1, c_in_p1, c_out_p1, bput1, net = 'agg', \
                           nn_val = 'cont', nn_dt = coa_dt, display_time = False)

    norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    contour = plt.contour(x, y, z, [0], colors='black')
    plt.clabel(contour, inline=True, fontsize=5)
    plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
               origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
    plt.plot(x, stockFwd_coa_p1[:-1,path], 'go-', color = 'magenta', linewidth=1)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+\
              str(np.round(np.mean(r_agg_p1),4)) + ' Payoff ' + \
              str(np.round(r_agg_p1[path],4)) + ' Stop: ' + \
              str(stop_agg_p1[path]))
    plt.savefig('Boundary-agg-NN-path-'+str(path)+'-dt-'+str(dt)+'-same-path-forensics-time.png', dpi=1000)
    plt.clf()

    # Training Data 
    (x,y,z) = NN_bound_neo(NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, net = 'agg', \
                           nn_val = 'cont', nn_dt = coa_dt, display_time = False)

    norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    contour = plt.contour(x, y, z, [0], colors='black')
    plt.clabel(contour, inline=True, fontsize=5)
    plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
               origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
    plt.plot(x, stockFwd_coa_p1[:-1,path], 'go-', color = 'magenta', linewidth=1)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+\
              str(np.round(np.mean(r_agg_data_p1),4)) + ' Payoff ' + \
              str(np.round(r_agg_data_p1[path],4)) + ' Stop: ' + \
              str(stop_agg_data_p1[path]))
    plt.savefig('Boundary-agg-NN-path-'+str(path)+'-dt-'+str(dt)+'-data-forensics-time.png', dpi=1000)
    plt.clf()
    print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,2), 'sec')


#########################
#### EXAMINING PATHS ####
#########################

## Testing 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_p1 = stock_sim(100000, bput1)
# Coarse Test Grid
coa_dt = 0.2
model_update(bput1, dt = coa_dt)
stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, coa_dt) 
(r_agg_coa_data_p1, stop_agg_coa_data_p1) = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_data_p1, c_in_p1, c_out_p1, \
                         val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)

### Path 8 
# Coarse Test Grid 
path = 8
dt = 0.2
model_update(bput1, dt = dt)
stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, dt) 

t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_coa_p1[:,path])
plt.plot(t, payoff(stockFwd_coa_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

# Fine Test Grid 
dt = 0.025
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_fine_p1[:,path])
plt.plot(t, payoff(stockFwd_fine_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

### Path 44 
path = 44
dt = 0.2
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_coa_p1[:,path])
plt.plot(t, payoff(stockFwd_coa_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

# Fine Test Grid 
dt = 0.025
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_fine_p1[:,path])
plt.plot(t, payoff(stockFwd_fine_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

### Path 4788 
path = 4788
dt = 0.2
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_coa_p1[:,path])
plt.plot(t, payoff(stockFwd_coa_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

# Fine Test Grid 
dt = 0.025
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_fine_p1[:,path])
plt.plot(t, payoff(stockFwd_fine_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

### Path 4789
path = 4789
dt = 0.2
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_coa_p1[:,path])
plt.plot(t, payoff(stockFwd_coa_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

# Fine Test Grid 
dt = 0.025
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_fine_p1[:,path])
plt.plot(t, payoff(stockFwd_fine_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

### Path 2950
path = 2950
# Fine Test Grid 
dt = 0.025
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_fine_p1[:,path])
plt.plot(t, payoff(stockFwd_fine_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

### Path Random
path = 75435
# Fine Test Grid 
dt = 0.025
t = np.arange(dt, bput1['T'] + dt, dt)
plt.plot(t, stockFwd_fine_p1[:,path])
plt.plot(t, payoff(stockFwd_fine_p1[:,path], bput1))
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '+str(np.round(r_agg_coa_data_p1[path],4))+' Stop '+str(stop_agg_coa_data_p1[path]))
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(['Stock', 'Payoff'])
plt.savefig('Path-'+str(path)+'-dt-'+str(dt)+'-forensics-time.png', dpi=1000)

# Sequence of Neural Networks
path = 44
(x,y,z) = NN_bound_neo(NN_seq_coa_p1, c_in_p1, c_out_p1, bput1, net = 'seq', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.plot(x, stockFwd_coa_p1[:4,path],'go-', color = 'red', linewidth=1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Path '+str(path)+' dt: '+str(dt)+' Price: '++str(np.round(r_seq_coa_p1[path],2)))
plt.savefig('Basket-Put-Boundary-seq-NN-path-8-forensics-time.png', dpi=1000)

# Aggregate Neural Network 
# Same Paths 
(x,y,z) = NN_bound_neo(NN_agg_coa_p1, c_in_p1, c_out_p1, bput1, net = 'agg', nn_val = 'cont', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.plot(x, stockFwd_coa_p1[:4,8],'go-', color = 'red', linewidth=1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(r_agg_coa_p1[8],4)))
plt.savefig('Basket-Put-Boundary-agg-NN-same-path-8-forensics-time.png', dpi=1000)
plt.clf()

# Training Data 
(x,y,z) = NN_bound_neo(NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, net = 'agg', nn_val = 'cont', display_time = False)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.plot(x, stockFwd_coa_p1[:9,8],'go-', color = 'red', linewidth=1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(r_agg_coa_data_p1[8],2)))
plt.savefig('Basket-Put-Boundary-agg-NN-data-path-8-forensics-time.png', dpi=1000)
'''  

#######################
#### 2D - Max Call ####
#######################

mcall2 = {'dim': 2, 'K': 100, 'x0': np.repeat(100, 2), 'sigma': np.repeat(0.2,2), 
         'r': 0.05, 'div': np.repeat(0,2), 'T': 1, 'dt': 0.05, 'payoff.func': 'maxi.call.payoff'}

# Initializing -- Fine Time Grid
model_update(mcall2, dt = 0.025)
np.random.seed(15)
torch.manual_seed(18)
stock_fine_m2 = stock_sim(100000, mcall2)
(c_in_m2, c_out_m2) = scaler(stock_fine_m2, mcall2)

# Thin at coarse dt
coa_dt = 0.05
model_update(mcall2, dt = coa_dt)
stock_coa_m2 = stock_thin(stock_fine_m2, mcall2, coa_dt) # Coarse time grid

## Sequence of Neural Networks -- Coarse Time Grid
epnum = 50
np.random.seed(15)
torch.manual_seed(18)
(NN_seq_coa_m2, x_m2, y_m2) = NN_seq_train_neo(stock_coa_m2, mcall2, c_in_m2, c_out_m2, \
                                    theta = 'average', data = True, val = 'cont', \
                                    node_num = 25, epoch_num = epnum, display_time=True)

## Testing 
# Initializing Fine Test Grid
model_update(mcall2, dt = 0.025)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_m2 = stock_sim(100000, mcall2)
coa_dt = 0.05
model_update(mcall2, dt = coa_dt)
stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, coa_dt) 

(r_seq_coa_m2, stop_seq_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'seq', NN_seq_coa_m2, c_in_m2, c_out_m2, \
                         val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)

print('dt:', coa_dt)
print('Price - seq NN:', np.round(np.mean(r_seq_coa_m2), 4))
  
# Contours from the Sequence of Neural Networks
model_update(mcall2, dt = 0.05)
norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
contours = []

# t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
for t in [0.1]:
    start_time = time.time()
    (x,y,z) = NN_contour_neo(t, NN_seq_coa_m2, c_in_m2, c_out_m2, mcall2, net = 'seq', display_time = False)
        
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[-1], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
               origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
    plt.colorbar()
    plt.plot(np.array([80,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.plot(np.array([100,100]), np.array([80,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.title('2-D Max Call Contour at Time Step '+str(np.round(t, 4)))
    plt.xlabel('Stock 1 Price')
    plt.ylabel('Stock 2 Price')
    plt.savefig('Max-Call-seq-NN-Map-'+str(np.round(t, 4))+'.png', dpi=1000)
    plt.clf()
    print('Map:', np.round(t,3),'Time:', np.round(time.time()-start_time,2), 'sec')


### Stage 2    
'''
## Aggregate Neural Network -- Coarse Time Grid
epnum = 5
np.random.seed(16)
torch.manual_seed(16)
# Same Paths
NN_agg_coa_m2 = NN_aggregate_neo(mcall2, NN_seq_coa_m2, c_in_m2, c_out_m2, nn_val = 'cont', \
                            stock = stock_coa_m2, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
# Training Data
np.random.seed(16)
torch.manual_seed(16)
NN_agg_coa_data_m2 = NN_aggregate_neo(mcall2, NN_seq_coa_m2, c_in_m2, c_out_m2, nn_val = 'cont', \
                            data = True, x = x_m2, y = y_m2, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
## Testing 
# Initializing Fine Test Grid
model_update(mcall2, dt = 0.025)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_m2 = stock_sim(100000, mcall2)

# [0.2, 0.1, 0.05, 0.025]
for dt in [0.2, 0.1, 0.05, 0.025]:
    start_time = time.time()
    model_update(mcall2, dt = dt)
    stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, dt) # Coarse Test Grid
    # Same Paths
    (r_agg_m2, stop_agg_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_m2, c_in_m2, c_out_m2, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
        
    # Training Data 
    (r_agg_data_m2, stop_agg_data_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_data_m2, c_in_m2, c_out_m2, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
     
    print('dt:', dt)
    # Same Paths  
    print('Price - agg NN - Path:', np.round(np.mean(r_agg_m2), 4))
    print('Avg Std agg NN - Path:', np.round(np.std(r_agg_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))
    # Training Data 
    print('Price - agg NN - Data:', np.round(np.mean(r_agg_data_m2), 4))
    print('Avg Std agg NN - Data:', np.round(np.std(r_agg_data_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))
    print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,2), 'sec')
    
###########################
#### FORENSICS & PLOTS ####
###########################
    
### Monitoring differences
np.amax(stop_agg_data_m2 - stop_agg_m2)
np.where((stop_agg_data_m2 - stop_agg_m2) == np.amax(stop_agg_data_m2 - stop_agg_m2))[0][:20]
np.amin(stop_agg_data_m2 - stop_agg_m2)
np.where((stop_agg_data_m2 - stop_agg_m2) == np.amin(stop_agg_data_m2 - stop_agg_m2))[0]
## Max differences 
# Paths 168, 298, 310, 511,  558,  563,  593,  728,  765,  779
# Min differences
## Paths 33781, 56589
# Interesting paths
# 27007, 77474

## Paths
path = 74820
model_update(mcall2, dt = 0.025)
t_steps = np.arange(mcall2['dt'], mcall2['T']+mcall2['dt'], mcall2['dt'])
plt.plot(t_steps, stockFwd_fine_m2[:, path])
plt.axhline(y = 100, color='black', linestyle='dashed', linewidth=1)
plt.axvline(x = stop_agg_m2[path], color = 'b', label = 'path', \
            linestyle='dashed', linewidth=1)
plt.axhline(y = mcall2['K']+r_agg_m2[path]*np.exp(mcall2['r']*stop_agg_m2[path]), color = 'b', label = 'data', \
            linestyle='dashed', linewidth=1)
plt.axvline(x = stop_agg_data_m2[path], color = 'm', label = 'data', \
            linestyle='dashed', linewidth=1)
plt.axhline(y = mcall2['K']+r_agg_data_m2[path]*np.exp(mcall2['r']*stop_agg_data_m2[path]), \
            color = 'm', label = 'data', linestyle='dashed', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Stock Value')
plt.title('Path: '+str(path)+' Stop Path: '+str(stop_agg_m2[path])+' Stop Data: '\
          +str(stop_agg_data_m2[path]))
plt.savefig('Loop-1-Path-'+str(path)+'-stop-data-'+str(np.round(stop_agg_data_m2[path],3))+\
            '-stop-path-'+str(np.round(stop_agg_m2[path],3))+'.png', dpi=1000)

plt.scatter(stop_agg_m2[:500], stop_agg_data_m2[:500], s=1, c='black', marker='o')
plt.plot([0, 1], [0, 1], color = 'red', linewidth=0.5)
plt.xlabel('Path')
plt.ylabel('Data')
    
t_inc = 0.01
p = []
for x in range(160,190,5):
    for y in range(190,195,5):
        pos = np.array([x, y])
        (x_data, y_data) = NN_time_neo(pos, NNagg, c_in_m2, c_out_m2, mcall2, \
                               nn_val = 'cont', nn_dt = coa_dt, t_inc=t_inc, display_time = False)
        p.append(list(pos))
        plt.plot(x_data, y_data, linewidth=1)
plt.plot(x_data, np.repeat(0, 100), color='black', linestyle='dashed', linewidth=1)
plt.legend(p)
plt.xlabel('Time')
plt.ylabel('Timing Value')
plt.title('Timing Values dt: '+str(t_inc)+' Loop 2 NN: '+str(0.025))
plt.savefig('Timing-values-dt-'+str(coa_dt)+'-pos-list-160-190-at-100-opposite.png', dpi=1000)
'''

#######################
#### 5D - Max Call ####
#######################

mcall5 = {'dim': 5, 'K': 100, 'x0': np.repeat(70, 5), 
         'sigma': np.array([0.08,0.16,0.24,0.32,0.4]), 
         'r': 0.05, 'div': np.repeat(0.1, 5), 'T': 3, 'dt': 1/3, 
         'payoff.func': 'maxi.call.payoff'}

# Initializing -- Fine Time Grid
model_update(mcall5, dt = 1/3)
np.random.seed(15)
torch.manual_seed(15)
stock_fine_m5 = stock_sim(50000, mcall5)
(c_in_m5, c_out_m5) = scaler(stock_fine_m5, mcall5)

# Thin at coarse dt
coa_dt = 1/3
model_update(mcall5, dt = coa_dt)
stock_coa_m5 = stock_thin(stock_fine_m5, mcall5, coa_dt) # Coarse time grid

## Sequence of Neural Networks -- Coarse Time Grid
epnum = 5
np.random.seed(15)
torch.manual_seed(15)
(NN_seq_coa_m5, x_m5, y_m5) = NN_seq_train_neo(stock_coa_m5, mcall5, c_in_m5, c_out_m5, \
                                    theta = 'average', data = True, val = 'cont', \
                                    node_num = 25, epoch_num = epnum, display_time=True)

## Testing 
# Initializing Fine Test Grid
model_update(mcall5, dt = 1/3)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_m5 = stock_sim(100000, mcall5)
coa_dt = 0.2
model_update(mcall5, dt = coa_dt)
stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall, coa_dt) 

(r_seq_coa_m5, stop_seq_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'seq', NN_seq_coa_m5, c_in_m5, c_out_m5, \
                         val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)

print('dt:', coa_dt)
print('Price - seq NN:', np.round(np.mean(r_seq_coa_m5), 4))


#########################
#### 2D - Basket Put ####
#########################

bput2 = {'dim': 2, 'K': 40, 'x0': np.repeat(40, 2), 'sigma': np.repeat(0.2,2), 
         'r': 0.06, 'div': np.repeat(0, 2), 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'}

#######################
#### 3D - Max Call ####
#######################

mcall3 = {'dim': 3, 'K': 100, 'x0': np.repeat(90, 3), 'sigma': np.repeat(0.2,3), 
         'r': 0.05, 'div': np.repeat(0.1, 3), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

### Stage 2
'''
## Aggregate Neural Network -- Coarse Time Grid
epnum = 5
np.random.seed(16)
torch.manual_seed(16)
# Same Paths
NN_agg_coa_m5 = NN_aggregate_neo(mcall5, NN_seq_coa_m5, c_in_m5, c_out_m5, nn_val = 'cont', \
                            stock = stock_coa_m5, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
# Training Data
np.random.seed(16)
torch.manual_seed(16)
NN_agg_coa_data_m5 = NN_aggregate_neo(mcall5, NN_seq_coa_m5, c_in_m5, c_out_m5, nn_val = 'cont', \
                            data = True, x = x_m5, y = y_m5, node_num = 25, batch_num = 64, \
                            epoch_num = epnum, display_time=True)
    
## Testing 
# Initializing Fine Test Grid
model_update(mcall5, dt = 1/3)
np.random.seed(18)
torch.manual_seed(18)
stockFwd_fine_m5 = stock_sim(100000, mcall5)

for dt in [1/3]:
    start_time = time.time()
    model_update(mcall5, dt = dt)
    stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall5, dt) # Coarse Test Grid
    # Same Paths
    (r_agg_m5, stop_agg_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_m5, c_in_m5, c_out_m5, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
        
    # Training Data 
    (r_agg_data_m5, stop_agg_data_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_data_m5, c_in_m5, c_out_m5, \
                             val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
     
    print('dt:', dt)
    # Same Paths  
    print('Price - agg NN - Path:', np.round(np.mean(r_agg_m5), 4))
    print('Avg Std agg NN - Path:', np.round(np.std(r_agg_m5, ddof = 1)/np.sqrt(len(stockFwd_coa_m5[0])), 4))
    # Training Data 
    print('Price - agg NN - Data:', np.round(np.mean(r_agg_data_m5), 4))
    print('Avg Std agg NN - Data:', np.round(np.std(r_agg_data_m5, ddof = 1)/np.sqrt(len(stockFwd_coa_m5[0])), 4))
    print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,5), 'sec')
'''
