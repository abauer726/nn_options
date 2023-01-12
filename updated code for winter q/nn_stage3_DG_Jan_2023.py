# Diagnostics -- January 2023

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


#######################
#### 2D - Max Call ####
#######################

mcall2 = {'dim': 2, 'K': 100, 'x0': np.repeat(100, 2), 'sigma': np.repeat(0.2,2), 
         'r': 0.05, 'div': np.repeat(0.1,2), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

### STAGE 1 ###

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
#(x_qmc_m2, time_val_m2, lr_m2, dif_m2, NN_max_m2) = NN_loop_v15(loops, loop_size, mcall2, \
#        NN_agg_m2, c_in_m2, c_out_m2, nn_dt = coa_dt, \
#        stock_check = stock_valid_m2, ci = 0.99, epoch_num = epnum_s3, display_time = True)

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

### Diagnostics ###
'''
dif = r_agg_stage3_m2 - r_max_stage3_m2
sorted_arr = sorted(zip(dif, range(len(dif))))

sorted_elements, indices = zip(*sorted_arr)
sorted_elements[:10]
indices[:10]

path = 87572
t_steps = np.arange(mcall2['dt'], mcall2['T']+mcall2['dt'], mcall2['dt'])
plt.plot(t_steps, stockFwd_fine_m2[:, path])
plt.axhline(y = 100, color='black', linestyle='dashed', linewidth=1)
plt.axvline(x = stop_stage3_m2[path], color = 'r', label = 'Stage 3 Final', \
            linestyle='dashed', linewidth=1)
plt.axhline(y = mcall2['K']+r_agg_stage3_m2[path]*np.exp(mcall2['r']*stop_stage3_m2[path]), \
            color = 'r', linestyle='dashed', linewidth=1)
plt.axvline(x = stop_max_stage3_m2[path], color = 'm', label = 'Stage 3 Max', \
            linestyle='dashed', linewidth=1)
plt.axhline(y = mcall2['K']+r_max_stage3_m2[path]*np.exp(mcall2['r']*stop_max_stage3_m2[path]), \
            color = 'm', linestyle='dashed', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Stock Value')
plt.legend()
plt.title('Path: '+str(path)+' Stop Stage 3 Final: '+str(np.round(stop_stage3_m2[path],3))+' Stop Stage 3 Max: '\
          +str(np.round(stop_max_stage3_m2[path],3)))
plt.savefig('Diff-Stage-3-Final-vs-Max-Path-'+str(path)+'-stop-final-'+str(np.round(stop_stage3_m2[path],3))+\
            '-stop-max-'+str(np.round(stop_max_stage3_m2[path],3))+'.png', dpi=1000)

plt.scatter(r_agg_stage2_dif_m2[:5000], r_agg_stage3_dif_m2[:5000], s=0.5, c='black', marker='o')
plt.plot([-0.2, 10], [-0.2, 10], color = 'red', linewidth=0.5)
plt.xlabel('Stage 2')
plt.ylabel('Stage 3')
plt.title('Stage 2 vs. Stage 3 Path Payoffs')
plt.savefig('MC-Stage-2-vs-3-loops-'+str(loops)+'-loop-size-'+str(loop_size)+'.png', dpi=1000)
plt.clf()


r_agg_stage2_itm_m2 = np.delete(r_agg_stage2_m2, np.where(r_agg_stage2_m2 == 0))
r_agg_stage3_itm_m2 = np.delete(r_agg_stage3_m2, np.where(r_agg_stage3_m2 == 0))

r_agg_stage2_dif_m2 = np.delete(r_agg_stage2_m2, np.where(r_agg_stage3_m2 == r_agg_stage2_m2))
r_agg_stage3_dif_m2 = np.delete(r_agg_stage3_m2, np.where(r_agg_stage3_m2 == r_agg_stage2_m2))

dif = r_agg_stage3_dif_m2-r_agg_stage2_dif_m2
sorted_arr = sorted(zip(dif, range(len(dif))))

sorted_elements, indices = zip(*sorted_arr)
sorted_elements[:10]
indices[:10]

dif2 = r_agg_stage3_m2 - r_agg_stage2_m2
sorted_arr = sorted(zip(dif2, range(len(dif2))))

sorted_elements, indices = zip(*sorted_arr)
sorted_elements[-10:-1]
indices[-10:-1]
# 53792, 63938, 81813, 65375, 71099, 46032, 99651, 56986, 11069, 67486
# 95004, 66820, 91837, 83511, 71255, 92423, 7860, 34347, 6857, 1125
# 25340, 72160, 51560, 20105, 65040, 54886, 54133, 97313, 98703

path = 20105
t_steps = np.arange(mcall2['dt'], mcall2['T']+mcall2['dt'], mcall2['dt'])
plt.plot(t_steps, stockFwd_fine_m2[:, path])
plt.axhline(y = 100, color='black', linestyle='dashed', linewidth=1)
plt.axvline(x = stop_stage2_m2[path], color = 'b', label = 'Stage 2', \
            linestyle='dashed', linewidth=1)
plt.axhline(y = mcall2['K']+r_agg_stage2_m2[path]*np.exp(mcall2['r']*stop_stage2_m2[path]), \
            color = 'b', linestyle='dashed', linewidth=1)
plt.axvline(x = stop_stage3_m2[path], color = 'm', label = 'Stage 3', \
            linestyle='dashed', linewidth=1)
plt.axhline(y = mcall2['K']+r_agg_stage3_m2[path]*np.exp(mcall2['r']*stop_stage3_m2[path]), \
            color = 'm', linestyle='dashed', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Stock Value')
plt.legend()
plt.title('Path: '+str(path)+' Stop Stage 2: '+str(np.round(stop_stage2_m2[path],3))+' Stop Stage 3: '\
          +str(np.round(stop_stage3_m2[path],3)))
plt.savefig('Diff-Stage-2-vs-3-Path-'+str(path)+'-stop-stage3-'+str(np.round(stop_stage3_m2[path],3))+\
            '-stop-path-'+str(np.round(stop_stage2_m2[path],3))+'.png', dpi=1000)

plt.scatter(r_agg_stage2_dif_m2[:5000], r_agg_stage3_dif_m2[:5000], s=0.5, c='black', marker='o')
plt.plot([-0.2, 10], [-0.2, 10], color = 'red', linewidth=0.5)
plt.xlabel('Stage 2')
plt.ylabel('Stage 3')
plt.title('Stage 2 vs. Stage 3 Path Payoffs')
plt.savefig('MC-Stage-2-vs-3-loops-'+str(loops)+'-loop-size-'+str(loop_size)+'.png', dpi=1000)
plt.clf()

# Contours of the Sequence of Neural Networks
model_update(mcall2, dt = 1/3)
norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
contours = []

# np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
# Contours 60 - 260
# points = 1000
for t in np.arange(mcall2['dt'], mcall2['T'], mcall2['dt']):
    start_time = time.time()
    ## Selecting points to display
    # loc = np.where(x_r_m2.transpose()[0] == t)[0][:points]
    # x_sample_m2 = []
    # for i in loc:
    #     if (x_r_m2[i][1:] >= 80).all() and (x_r_m2[i][1:] <= 180).all():
    #         x_sample_m2.append(x_r_m2[i][1:])
    # pts = np.asarray(x_sample_m2).transpose()
    ## Contour
    (x,y,z) = NN_contour_neo(t, NN_agg_m2, c_in_m2, c_out_m2, mcall2, \
                             net = 'agg', nn_val = 'cont', nn_dt = coa_dt, \
                             down = 0.6, up = 2.0, inc = 0.25, display_time = False)
    
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[-1], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[60, 200, 60, 200], \
               origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
    plt.colorbar()
    plt.plot(np.array([60,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.plot(np.array([100,100]), np.array([60,100]), linewidth=1, color= 'black', linestyle='dashdot')
    # plt.scatter(pts[0], pts[1], s = 1.5, marker='o', c = 'black')
    plt.title('Stage 3 - Map: '+str(np.round(t, 2))+' Price: '+str(np.round(np.mean(r_agg_stage3_m2),4)))
    plt.savefig('MC-Stage-3-ReLU-NN-Final-Map-'+str(np.round(t, 2))+'-loops-'+str(loops)+'-loop-size-'+str(loop_size)+'-wide.png', dpi=1000)
    plt.clf()
    print('Stage 3 Map:', np.round(t,3), 'Price:', np.round(np.mean(r_agg_stage3_m2), 4), \
          'Time:', np.round(time.time()-start_time,2), 'sec')

for t in np.arange(mcall2['dt'], mcall2['T'], mcall2['dt']):
    start_time = time.time()
    ## Selecting points to display
    # loc = np.where(x_r_m2.transpose()[0] == t)[0][:points]
    # x_sample_m2 = []
    # for i in loc:
    #     if (x_r_m2[i][1:] >= 80).all() and (x_r_m2[i][1:] <= 180).all():
    #         x_sample_m2.append(x_r_m2[i][1:])
    # pts = np.asarray(x_sample_m2).transpose()
    ## Contour
    (x,y,z) = NN_contour_neo(t, NN_max_m2, c_in_m2, c_out_m2, mcall2, \
                             net = 'agg', nn_val = 'cont', nn_dt = coa_dt, \
                             down = 0.6, up = 2.0, inc = 0.25, display_time = False)
    
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[-1], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[60, 200, 60, 200], \
               origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
    plt.colorbar()
    plt.plot(np.array([60,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
    plt.plot(np.array([100,100]), np.array([60,100]), linewidth=1, color= 'black', linestyle='dashdot')
    # plt.scatter(pts[0], pts[1], s = 1.5, marker='o', c = 'black')
    plt.title('Stage 3 - Map: '+str(np.round(t, 2))+' Price: '+str(np.round(np.mean(r_max_stage3_m2),4)))
    plt.savefig('MC-Stage-3-ReLU-NN-Max-Map-'+str(np.round(t, 2))+'-loops-'+str(loops)+'-loop-size-'+str(loop_size)+'-wide.png', dpi=1000)
    plt.clf()
    print('Stage 3 Map:', np.round(t,3), 'Price:', np.round(np.mean(r_max_stage3_m2), 4), \
          'Time:', np.round(time.time()-start_time,2), 'sec')
'''
        
#######################
#### 3D - Max Call ####
#######################

mcall3 = {'dim': 3, 'K': 100, 'x0': np.repeat(90, 3), 'sigma': np.repeat(0.2,3), 
         'r': 0.05, 'div': np.repeat(0.1, 3), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

# Initializing -- Fine Time Grid
model_update(mcall3, dt = 1/3)
np.random.seed(15)
tf.random.set_seed(15)
stock_fine_m3 = stock_sim(100000, mcall3)
(c_in_m3, c_out_m3) = scaler(stock_fine_m3, mcall3)

# Thin at coarse dt
coa_dt = 1/3
model_update(mcall3, dt = coa_dt)
stock_coa_m3 = stock_thin(stock_fine_m3, mcall3, coa_dt) # Coarse time grid

## Sequence of Neural Networks -- Coarse Time Grid
epnum = 5
lr = 0.001
b1 = 0.95
b2 = 0.999
opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
np.random.seed(15)
tf.random.set_seed(15)
(NN_seq_coa_m3, x_m3, y_m3) = NN_seq_train_neo(stock_coa_m3, mcall3, c_in_m3, c_out_m3, \
                                theta = 'average', data = True, val = 'cont', \
                                node_num = 25, epoch_num = epnum, optim = opt, display_time=False)

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(mcall3, dt = 1/3)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_m3 = stock_sim(100000, mcall3)
model_update(mcall3, dt = coa_dt)
stockFwd_coa_m3 = stock_thin(stockFwd_fine_m3, mcall3, coa_dt)

# Computing the option price
r_seq_m3 = NN_payoff_neo(0, stockFwd_coa_m3, mcall3, 'seq', NN_seq_coa_m3, c_in_m3, \
                              c_out_m3, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
print('Stage 1 Price:', np.round(np.mean(r_seq_m3), 4))

### STAGE 2 ###

# Shuffling data
np.random.seed(0)
merge = np.append(x_m3.transpose(), y_m3.transpose()).reshape(mcall3['dim']+2, len(y_m3)).transpose()
np.random.shuffle(merge)
x_r_m3 = merge.transpose()[:-1].transpose()
y_r_m3 = merge.transpose()[-1].reshape(-1, 1) 

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(mcall3, dt = 1/3)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_m3 = stock_sim(100000, mcall3)
model_update(mcall3, dt = coa_dt)
stockFwd_coa_m3 = stock_thin(stockFwd_fine_m3, mcall3, coa_dt)

# Stage 2: Initializing the Aggregate Network
epnum = 5
lr = 0.001
b1 = 0.95
b2 = 0.999
opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
np.random.seed(16)
tf.random.set_seed(16)
NN_agg_m3 = NN_aggregate_neo(mcall3, NN_seq_coa_m3, c_in_m3, c_out_m3, \
                        nn_val = 'cont', data = True, x = x_r_m3, y = y_r_m3, \
                        node_num = 25, batch_num = 64, epoch_num = epnum, \
                        optim = opt, display_time=False)

# Computing the option price
(r_agg_stage2_m3, stop_stage2_m3) = NN_payoff_neo(0, stockFwd_coa_m3, mcall3, 'agg', NN_agg_m3, c_in_m3, \
                              c_out_m3, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, 
                              stop = True, display_time=False)
        
print('Stage 2 Price:', np.round(np.mean(r_agg_stage2_m3), 4))

### STAGE 3 ###

### Setting up the validating paths
# model_update(mcall3, dt = 1/3)
np.random.seed(99)
tf.random.set_seed(99)
stock_valid_m3 = stock_sim(10000, mcall3)

np.random.seed(16)
tf.random.set_seed(16)
loops = 40
loop_size = 1000
epnum_s3 = 5
# For specific learning rate
# optp = [0.001, 0.95, 0.999]

# Looping
# (x_qmc_m3, time_val_m3, lr_m3, dif_m3, NN_max_m3) = NN_loop_v15(loops, loop_size, mcall3, \
#         NN_agg_m3, c_in_m3, c_out_m3, nn_dt = coa_dt, \
#         stock_check = stock_valid_m3, ci = 0.99, epoch_num = epnum_s3, display_time = True)

NN_max_m3 = NN_loop_v17(loops, loop_size, mcall3, NN_agg_m3, c_in_m3, c_out_m3, \
                        nn_dt = coa_dt, stock_check = stock_valid_m3, epoch_num = epnum_s3, \
                        display_time = True)
    
## Testing 
# Final Loop Price
(r_agg_stage3_m3, stop_stage3_m3) = NN_payoff_neo(0, stockFwd_coa_m3, mcall3, 'agg', \
            NN_agg_m3, c_in_m3, c_out_m3, val = 'cont', nn_val = 'cont', \
            nn_dt = coa_dt, stop = True, display_time=False)
# Max Loop Price
(r_max_stage3_m3, stop_max_stage3_m3) = NN_payoff_neo(0, stockFwd_coa_m3, mcall3, 'agg', \
            NN_max_m3, c_in_m3, c_out_m3, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, \
            stop = True, display_time=False)
    
print('Stage 3 -Final- Price:', np.round(np.mean(r_agg_stage3_m3), 4))
print('Stage 3 -Max- Price:', np.round(np.mean(r_max_stage3_m3), 4))

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
tf.random.set_seed(15)
stock_fine_m5 = stock_sim(100000, mcall5)
(c_in_m5, c_out_m5) = scaler(stock_fine_m5, mcall5)

# Thin at coarse dt
coa_dt = 1/3
model_update(mcall5, dt = coa_dt)
stock_coa_m5 = stock_thin(stock_fine_m5, mcall5, coa_dt) # Coarse time grid

## Sequence of Neural Networks -- Coarse Time Grid
epnum = 5
lr = 0.001
b1 = 0.95
b2 = 0.999
opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
np.random.seed(15)
tf.random.set_seed(15)
(NN_seq_coa_m5, x_m5, y_m5) = NN_seq_train_neo(stock_coa_m5, mcall5, c_in_m5, c_out_m5, \
                                theta = 'average', data = True, val = 'cont', \
                                node_num = 25, epoch_num = epnum, optim = opt, display_time=False)

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(mcall5, dt = 1/3)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_m5 = stock_sim(100000, mcall5)
model_update(mcall5, dt = coa_dt)
stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall5, coa_dt)

# Computing the option price
r_seq_m5 = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'seq', NN_seq_coa_m5, c_in_m5, \
                              c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
print('Stage 1 Price:', np.round(np.mean(r_seq_m5), 4))

### STAGE 2 ###

# Shuffling data
np.random.seed(0)
merge = np.append(x_m5.transpose(), y_m5.transpose()).reshape(mcall5['dim']+2, len(y_m5)).transpose()
np.random.shuffle(merge)
x_r_m5 = merge.transpose()[:-1].transpose()
y_r_m5 = merge.transpose()[-1].reshape(-1, 1) 

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(mcall5, dt = 1/3)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_m5 = stock_sim(100000, mcall5)
model_update(mcall5, dt = coa_dt)
stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall5, coa_dt)

# Stage 2: Initializing the Aggregate Network
epnum = 5
lr = 0.001
b1 = 0.95
b2 = 0.999
opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
np.random.seed(16)
tf.random.set_seed(16)
NN_agg_m5 = NN_aggregate_neo(mcall5, NN_seq_coa_m5, c_in_m5, c_out_m5, \
                        nn_val = 'cont', data = True, x = x_r_m5, y = y_r_m5, \
                        node_num = 25, batch_num = 64, epoch_num = epnum, \
                        optim = opt, display_time=False)

# Computing the option price
(r_agg_stage2_m5, stop_stage2_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_m5, c_in_m5, \
                              c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, 
                              stop = True, display_time=False)
        
print('Stage 2 Price:', np.round(np.mean(r_agg_stage2_m5), 4))

### STAGE 3 ###

### Setting up the validating paths
# model_update(mcall5, dt = 1/3)
np.random.seed(99)
tf.random.set_seed(99)
stock_valid_m5 = stock_sim(10000, mcall5)

np.random.seed(16)
tf.random.set_seed(16)
loops = 40
loop_size = 1000
epnum_s3 = 5
# For specific learning rate
# optp = [0.001, 0.95, 0.999]

# Looping
# (x_qmc_m5, time_val_m5, lr_m5, dif_m5, NN_max_m5) = NN_loop_v15(loops, loop_size, mcall5, \
#         NN_agg_m5, c_in_m5, c_out_m5, nn_dt = coa_dt, \
#         stock_check = stock_valid_m5, ci = 0.99, epoch_num = epnum_s3, display_time = True)

NN_max_m5 = NN_loop_v17(loops, loop_size, mcall5, NN_agg_m5, c_in_m5, c_out_m5, \
                        nn_dt = coa_dt, stock_check = stock_valid_m5, epoch_num = epnum_s3, \
                        display_time = True)
    
## Testing 
# Final Loop Price
(r_agg_stage3_m5, stop_stage3_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', \
            NN_agg_m5, c_in_m5, c_out_m5, val = 'cont', nn_val = 'cont', \
            nn_dt = coa_dt, stop = True, display_time=False)
# Max Loop Price
(r_max_stage3_m5, stop_max_stage3_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', \
            NN_max_m5, c_in_m5, c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, \
            stop = True, display_time=False)
    
print('Stage 3 -Final- Price:', np.round(np.mean(r_agg_stage3_m5), 4))
print('Stage 3 -Max- Price:', np.round(np.mean(r_max_stage3_m5), 4))


###############################################################################
#### TRIALS ###################################################################
###############################################################################

time_val_m2 = np.asarray(time_val_m2, dtype=float)

np.random.seed(16)
tf.random.set_seed(16)

plt.scatter(x_qmc_m2.transpose()[0], x_qmc_m2.transpose()[1])

sort1 = (time_val_m2[1] - time_val_m2[0])/(time_val_m2[0]**2+1)
np.sort((time_val_m2[1] - time_val_m2[0])/(time_val_m2[0]**2+1))
plt.hist((time_val_m2[1] - time_val_m2[0])/(time_val_m2[0]**2+0.01))

np.percentile(sort1, 99) 

sort2 = np.abs(time_val_m2[0])

## Gaussian function
sort3 = np.exp(-0.5*((time_val_m2[1] - time_val_m2[0])**2)/2)

alpha = 0.25
sort4 = (time_val_m2[1] - time_val_m2[0])*np.exp(-alpha*np.abs(time_val_m2[0]))
plt.hist((time_val_m2[1] - time_val_m2[0])/np.exp(-alpha*np.abs(time_val_m2[0])))



sorted_arr = sorted(zip(sort1, range(len(sort1))))
sorted_arr = sorted(zip(sort2, range(len(sort2))))
sorted_arr = sorted(zip(sort3, range(len(sort3))))
sorted_arr = sorted(zip(sort4, range(len(sort4))))

# Unzip the sorted array and print the indices
sorted_elements, indices = zip(*sorted_arr)
sorted_elements[-100:-1]
indices[-100:-1]
(time_val_m2[0][4227], time_val_m2[1][4227])
(time_val_m2[0][1744], time_val_m2[1][1744])
(time_val_m2[0][6792], time_val_m2[1][6792])
(time_val_m2[0][2832], time_val_m2[1][2832])
(time_val_m2[0][313], time_val_m2[1][313])
x_qmc_m2[4227]
x_qmc_m2[1744]
x_qmc_m2.shape

inputs_num = 1000
x_sel = np.asanyarray([x_qmc_m2[i] for i in indices[-inputs_num:-1]], dtype=float)
plt.scatter(x_sel.transpose()[0], x_sel.transpose()[1], s=0.5)
plt.plot(np.array([min(x_sel.transpose()[0]),100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
plt.plot(np.array([100,100]), np.array([min(x_sel.transpose()[1]),100]), linewidth=1, color= 'black', linestyle='dashdot')
plt.xlabel('Stock 1')
plt.ylabel('Stock 2')
plt.title('Selecting '+str(inputs_num)+' Inputs for Active Learning')
plt.savefig('Active Learning-inputs-'+str(inputs_num)+'.png', dpi=1000)


###############################################################################

## Stability Metrics

time_val_m2 = np.asarray(time_val_m2, dtype=float)

# Metric -- Squared difference between timing values
l2_dif_m2 = []
l2_dif_m3 = []
l2_dif_m5 = []
# Square difference between timing values
l1_dif_m2 = []
l1_dif_m3 = []
l1_dif_m5 = []
for x in range(1, len(time_val_m2)):
    l2_dif_m2.append(np.mean(np.sqrt(((time_val_m2[x] - time_val_m2[x-1])**2).astype(float))))
    l2_dif_m3.append(np.mean(np.sqrt(((time_val_m3[x] - time_val_m3[x-1])**2).astype(float))))
    l2_dif_m5.append(np.mean(np.sqrt(((time_val_m5[x] - time_val_m5[x-1])**2).astype(float))))
    l1_dif_m2.append(np.mean(np.abs(time_val_m2[x] - time_val_m2[x-1])))
    l1_dif_m3.append(np.mean(np.abs(time_val_m3[x] - time_val_m3[x-1])))
    l1_dif_m5.append(np.mean(np.abs(time_val_m5[x] - time_val_m5[x-1])))
    
'''    
plt.scatter(lr_m2[1:], dif_m2, s=1, c='black', marker='o')
plt.axhline(y=0, color = 'red', linewidth=0.25)
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Change in Price')
plt.title('Change in Price vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-P-vs-LR.png', dpi=1000)
plt.clf()
'''

plt.scatter(lr_m2[1:], dif_m2, s=1, c='black', marker='o')
plt.axhline(y=0, color = 'red', linewidth=0.25)
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Change in Price')
plt.title('2D MC Change in Price vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-P-vs-LR-x-log-scale.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Log L2 Change in Timing Values')
plt.title('2D MC L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-xy-log-scale.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o')
plt.axhline(y=0, color = 'red', linewidth=0.25)
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('L2 Change in Timing Values')
plt.title('2D MC L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-x-log-scale.png', dpi=1000)
plt.clf()

'''
plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Log L2 Change in Timing Values')
plt.title('2D MC L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-y-log-scale.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o')
plt.axhline(y=0, color = 'red', linewidth=0.25)
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('L2 Change in Timing Values')
plt.title('2D MC L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR.png', dpi=1000)
plt.clf()
'''
#############
### All 3 ###
#############

plt.scatter(lr_m2, dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(lr_m3, dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(lr_m5, dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Change in Price')
plt.title('Change in Price vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-P-vs-LR-all.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2, dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(lr_m3, dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(lr_m5, dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Log L2 Change in Timing Values')
plt.title('Change in Price vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-P-vs-LR-x-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(lr_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(lr_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('Log L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-xy-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(lr_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(lr_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('Log Learning Rate')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-x-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(lr_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(lr_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Log L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-y-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(lr_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(lr_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(lr_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Learning Rate w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-LR-all.png', dpi=1000)
plt.clf()

plt.scatter(dif_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(dif_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(dif_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Change in Price')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Change in Price  w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-Delta-P-all.png', dpi=1000)
plt.clf()

plt.scatter(dif_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(dif_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(dif_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Change in Price')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Change in Price  w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-Delta-P-y-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(dif_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(dif_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(dif_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('Change in Price')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Change in Price  w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-Delta-P-x-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(np.abs(dif_m2[1:]), l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(np.abs(dif_m3[1:]), l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(np.abs(dif_m5[1:]), l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('Absolute Change in Price')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Change in Price  w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-Abs-Delta-P-x-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(dif_m2[1:], l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(dif_m3[1:], l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(dif_m5[1:], l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Change in Price')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Change in Price w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-Delta-P-xy-log-scale-all.png', dpi=1000)
plt.clf()

plt.scatter(np.abs(dif_m2[1:]), l2_dif_m2, s=1, c='black', marker='o', label='M2')
plt.scatter(np.abs(dif_m3[1:]), l2_dif_m3, s=1, c='red', marker='o', label='M3')
plt.scatter(np.abs(dif_m5[1:]), l2_dif_m5, s=1, c='blue', marker='o', label='M5')
plt.axhline(y=0, color = 'red', linewidth=0.25)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Absolute Change in Price')
plt.ylabel('L2 Change in Timing Values')
plt.title('L2 Change in Timing Values vs. Change in Price w. '+str(loops)+' Loops')
plt.savefig('MC-Stage-3-loops-'+str(loops)+'-Delta-T-vs-Abs-Delta-P-xy-log-scale-all.png', dpi=1000)
plt.clf()