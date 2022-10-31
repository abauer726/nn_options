# Stage 3 - Updating

from scaler import scaler
from stock_v5 import stock_sim
from stock_v5 import stock_thin
from nn_train_neo import NN_seq_train_neo
from nn_aggregate_v3 import NN_payoff_neo
from nn_aggregate_v3 import NN_aggregate_neo
#from nn_update_v5 import NN_loop_v5
from plots_neo import NN_bound_neo
from plots_neo import NN_contour_neo
from plots_neo import NN_time_neo
from model_update import model_update

## Libraries
import time
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf


#########################
#### 1D - Basket Put ####
#########################

bput1 = {'dim': 1, 'K': 40, 'x0': np.repeat(40, 1), 'sigma': np.repeat(0.2,1), 
         'r': 0.05, 'div': np.repeat(0, 1), 'T': 1, 'dt': 0.05, 'payoff.func': 'put.payoff'}


# Initializing -- Fine Time Grid
model_update(bput1, dt = 0.025)
np.random.seed(15)
tf.random.set_seed(15)
stock_fine_p1 = stock_sim(100000, bput1)
(c_in_p1, c_out_p1) = scaler(stock_fine_p1, bput1)

# Thin at coarse dt
coa_dt = 0.1
model_update(bput1, dt = coa_dt)
stock_coa_p1 = stock_thin(stock_fine_p1, bput1, coa_dt) # Coarse time grid

## Sequence of Neural Networks -- Coarse Time Grid
epnum = 5
beta_1_anna = 0.9999
lr_anna = 0.0001
opt = tf.keras.optimizers.Adam(learning_rate=lr_anna, beta_1=beta_1_anna, beta_2=0.999)
np.random.seed(15)
tf.random.set_seed(15)
(NN_seq_coa_p1, x_p1, y_p1) = NN_seq_train_neo(stock_coa_p1, bput1, c_in_p1, c_out_p1, \
                                theta = 'average', data = True, val = 'cont', node_num = 25, \
                                epoch_num = epnum, optim = opt, display_time=True)

########################################
### Part of Diagnostics -- Beginning ###    
########################################

# Shuffling data
np.random.seed(0)
merge = np.append(x_p1.transpose(), y_p1.transpose()).reshape(bput1['dim']+2, len(y_p1)).transpose()
np.random.shuffle(merge)
x_r_p1 = merge.transpose()[:-1].transpose()
y_r_p1 = merge.transpose()[-1].reshape(-1, 1) 

### Setting up the testing framework 
# Initializing Fine Test Grid
model_update(bput1, dt = 0.025)
np.random.seed(18)
tf.random.set_seed(18)
stockFwd_fine_p1 = stock_sim(100000, bput1)
nn_dt = coa_dt
model_update(bput1, dt = coa_dt)
stockFwd_coa_p1 = stock_thin(stockFwd_fine_p1, bput1, coa_dt)

# Selecting data size for each loop
data_size = 20000

# Stage 2: Initializing the Aggregate Network
epnum = 5
opt = tf.keras.optimizers.Adam(learning_rate=lr_anna, beta_1=beta_1_anna, beta_2=0.999)
np.random.seed(16)
tf.random.set_seed(16)
NN_agg_coa_data_p1 = NN_aggregate_neo(bput1, NN_seq_coa_p1, c_in_p1, c_out_p1, \
                        nn_val = 'cont', data = True, x = x_r_p1[:data_size], \
                        y = y_r_p1[:data_size], node_num = 25, batch_num = 64, \
                        epoch_num = epnum, optim = opt, display_time=False)

# Computing the option price
r_agg_data_p1 = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_data_p1, c_in_p1, \
                              c_out_p1, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
print('Loop: 0 Price:', np.round(np.mean(r_agg_data_p1), 4))

## Timing Values = Continuation Value - Immediate Payoff
time_val_p1 = []

## Boundary Plot
# Aggregate Network
points = 250
(x,y,z) = NN_bound_neo(NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, net = 'agg', 
                       nn_val = 'cont', nn_dt = coa_dt, display_time = False)

time_val_p1.append(z)

norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
           origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
plt.colorbar()
plt.plot(np.arange(0, bput1['dt'] + bput1['T'], bput1['dt']), np.repeat(bput1['K'], 
        int(bput1['T']/bput1['dt'])+1), linewidth=1, color= 'black', linestyle='dashdot')
plt.scatter(x_r_p1[:data_size].transpose()[0,:points], x_r_p1[:data_size].transpose()[1,:points], 
            s = 3, marker='o', c = 'black')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Loop 0 Boundary Plot - learning rate: '+str(lr_anna)+'-b1-'+str(beta_1_anna)+' Price: '+str(np.round(np.mean(r_agg_data_p1),4)))
plt.savefig('Put-Boundary-agg-NN-loop-0-learning rate-'+str(lr_anna)+'-paths-'+str(data_size)+'-points-'+str(points)+'.png', dpi=1000)
plt.clf()

## Stage 3: adding training paths 
    
for d in range(1, int(len(x_r_p1)/data_size)):
    x_stage2_p1 = x_r_p1[data_size*d:data_size*(d+1)]
    y_stage2_p1 = y_r_p1[data_size*d:data_size*(d+1)]

    # Scaling additional data inputs and outputs   
    input_train_scaled_agg = []
    for j in range(bput1['dim']+1):
        input_train = x_stage2_p1.transpose()[j].reshape(-1, 1)
        input_train_scaled_agg.append(c_in_p1[j].transform(input_train))
    x_ = np.stack(input_train_scaled_agg).transpose()[0]
    y_ = c_out_p1.transform(y_stage2_p1.reshape(-1,1))

    # Neural Network updating parameteres
    epnum = 5
    batch_num = 64
    lossfct = 'mean_squared_error'
    # optim = 'adam'
    # optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_anna, beta_1=beta_1_anna, beta_2=0.999)
    
    NN_agg_coa_data_p1.compile(optimizer = opt, loss = lossfct)
    NN_agg_coa_data_p1.fit(x_, y_, epochs = epnum, batch_size = batch_num, verbose = 0)

    # Computing the option price
    r_agg_data_p1 = NN_payoff_neo(0, stockFwd_coa_p1, bput1, 'agg', NN_agg_coa_data_p1, c_in_p1, \
                              c_out_p1, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
    print('Loop:', d, 'Price:', np.round(np.mean(r_agg_data_p1), 4))

    ## Boundary Plot
    # Sequence of Neural Networks 
    (x,y,z) = NN_bound_neo(NN_agg_coa_data_p1, c_in_p1, c_out_p1, bput1, net = 'agg', 
                           nn_val = 'cont', nn_dt = coa_dt, display_time = False)
    
    time_val_p1.append(z)
    norm_put = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    contour = plt.contour(x, y, z, [0], colors='black')
    plt.clabel(contour, inline=True, fontsize=5)
    plt.imshow(np.array(z, dtype = float), extent = [0, 1, 30, 41],  aspect='auto', \
               origin='lower', cmap='Spectral', norm = norm_put, alpha=1)
    plt.colorbar()
    plt.plot(np.arange(0, bput1['dt'] + bput1['T'], bput1['dt']), np.repeat(bput1['K'], 
            int(bput1['T']/bput1['dt'])+1), linewidth=1, color= 'black', linestyle='dashdot')
    plt.scatter(x_stage2_p1.transpose()[0,:points], x_stage2_p1.transpose()[1,:points], 
                s = 3, marker='o', c = 'black')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Loop '+str(d)+' Boundary Plot - dt: '+str(coa_dt)+' Price: '+str(np.round(np.mean(r_agg_data_p1),4))+'-b1-'+str(beta_1_anna)+'-LR-'+str(lr_anna))
    plt.savefig('Put-Boundary-agg-NN-loop-'+str(d)+'-dt-'+str(coa_dt)+'-paths-'+str(data_size)+'-points-'+str(points)+'.png', dpi=1000)
    plt.clf()
    

## Stability Metrics

np.abs(np.asarray(time_val_p1)[1] - np.asarray(time_val_p1)[0])
# Metric -- Squared difference between timing values
sq_dif_p1 = []
# Square difference between timing values
abs_dif_p1 = []
for x in range(1, int(len(x_r_p1)/data_size)):
    sq_dif_p1.append(np.sum((np.asarray(time_val_p1)[x] - np.asarray(time_val_p1)[x-1])**2))
    abs_dif_p1.append(np.sum(np.abs(np.asarray(time_val_p1)[x] - np.asarray(time_val_p1)[x-1])))

plt.plot(range(1, len(sq_dif_p1)+1), np.asarray(sq_dif_p1)/sq_dif_p1[0])
plt.plot(range(1, len(sq_dif_p1)+1), np.repeat(1,len(sq_dif_p1)), linewidth=1, color= 'black', linestyle='dashdot')
plt.xlabel('Level')
plt.ylabel('Loop')
plt.title('Put - Squared Diff - Paths: '+str(data_size)+'-B1-'+str(beta_1_anna)+'-LR-'+str(lr_anna))
plt.savefig('Put-Stability-sq-diff-dt-'+str(coa_dt)+'-paths-'+str(data_size)+'-B1-'+str(beta_1_anna)+'-LR-'+str(lr_anna)+'.png', dpi=1000)
plt.clf()

plt.plot(range(1, len(abs_dif_p1)+1), np.asarray(abs_dif_p1)/abs_dif_p1[0])
plt.plot(range(1, len(abs_dif_p1)+1), np.repeat(1,len(abs_dif_p1)), linewidth=1, color= 'black', linestyle='dashdot')
plt.xlabel('Level')
plt.ylabel('Loop')
plt.title('Put - Absolute Diff - Paths: '+str(data_size)+'-B1-'+str(beta_1_anna)+'-LR-'+str(lr_anna))
plt.savefig('Put-Stability-abs-diff-dt-'+str(coa_dt)+'-paths-'+str(data_size)+'-B1-'+str(beta_1_anna)+'-LR-'+str(lr_anna)+'.png', dpi=1000)
plt.clf()
    
#######################
#### 2D - Max Call ####
#######################

# mcall2 = {'dim': 2, 'K': 100, 'x0': np.repeat(100, 2), 'sigma': np.repeat(0.2,2), 
#          'r': 0.05, 'div': np.repeat(0.1,2), 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

# # Initializing -- Fine Time Grid
# model_update(mcall2, dt = 1/3)
# np.random.seed(15)
# tf.random.set_seed(15)
# stock_fine_m2 = stock_sim(100000, mcall2)
# (c_in_m2, c_out_m2) = scaler(stock_fine_m2, mcall2)

# # Thin at coarse dt
# coa_dt = 1/3
# model_update(mcall2, dt = coa_dt)
# stock_coa_m2 = stock_thin(stock_fine_m2, mcall2, coa_dt) # Coarse time grid

# ## Sequence of Neural Networks -- Coarse Time Grid
# epnum = 5
# opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
# np.random.seed(15)
# tf.random.set_seed(15)
# (NN_seq_coa_m2, x_m2, y_m2) = NN_seq_train_neo(stock_coa_m2, mcall2, c_in_m2, c_out_m2, \
#                                 theta = 'average', data = True, val = 'cont', \
#                                 node_num = 25, epoch_num = epnum, optim = opt, display_time=True)

# ### Setting up the testing framework 
# # Initializing Fine Test Grid
# model_update(mcall2, dt = 1/3)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m2 = stock_sim(100000, mcall2)
# model_update(mcall2, dt = coa_dt)
# stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, coa_dt)

# # Computing the option price
# r_seq_m2 = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'seq', NN_seq_coa_m2, c_in_m2, \
#                               c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
# print('Seq Price:', np.round(np.mean(r_seq_m2), 4))

# # Contours of the Sequence of Neural Networks
# model_update(mcall2, dt = 1/3)
# norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
# contours = []

# # t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
# for t in [1]:
#     start_time = time.time()
#     ## Contour
#     (x,y,z) = NN_contour_neo(t, NN_seq_coa_m2, c_in_m2, c_out_m2, mcall2, net = 'seq', \
#                              down = 0.8, up = 1.8, display_time = False)
    
#     contours.append(plt.contour(x, y, z, [0], colors='black'))
#     plt.clabel(contours[-1], inline=True, fontsize=10)
#     plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
#                origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
#     plt.colorbar()
#     plt.plot(np.array([80,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
#     plt.plot(np.array([100,100]), np.array([80,100]), linewidth=1, color= 'black', linestyle='dashdot')
#     plt.title('Seq NN - Stage 1 - Map: '+str(np.round(t, 4))+' Loop: 0 Price: '+str(np.round(np.mean(r_seq_m2),4)))
#     plt.savefig('Max-Call-seq-NN-relu-Map-'+str(np.round(t, 4))+'.png', dpi=1000)
#     plt.clf()
#     print('Map:', np.round(t,3),'Time:', np.round(time.time()-start_time,2), 'sec')

# ########################################
# ### Part of Diagnostics -- Beginning ###    
# ########################################

# # Shuffling data
# np.random.seed(0)
# merge = np.append(x_m2.transpose(), y_m2.transpose()).reshape(mcall2['dim']+2, len(y_m2)).transpose()
# np.random.shuffle(merge)
# x_r_m2 = merge.transpose()[:-1].transpose()
# y_r_m2 = merge.transpose()[-1].reshape(-1, 1) 

# ### Setting up the testing framework 
# # Initializing Fine Test Grid
# model_update(mcall2, dt = 1/3)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m2 = stock_sim(100000, mcall2)
# model_update(mcall2, dt = coa_dt)
# stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, coa_dt)

# # Selecting data size for each loop
# data_size = 20000

# # Stage 2: Initializing the Aggregate Network
# epnum = 5
# opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
# np.random.seed(16)
# tf.random.set_seed(16)
# NN_agg_coa_data_m2 = NN_aggregate_neo(mcall2, NN_seq_coa_m2, c_in_m2, c_out_m2, nn_val = 'cont', \
#                         data = True, x = x_r_m2[:data_size], y = y_r_m2[:data_size], node_num = 25, batch_num = 64, \
#                         epoch_num = epnum, optim = opt, display_time=False)

# # Computing the option price
# r_agg_data_m2 = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_data_m2, c_in_m2, \
#                               c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
# print('Loop: 0 Price:', np.round(np.mean(r_agg_data_m2), 4))

# # Contours of the Aggregate Neural Network 
# model_update(mcall2, dt = 1/3)
# norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
# contours = []
# time_val_m2 = []
# points = 500

# # t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
# for t in [1]:
#     start_time = time.time()
#     ## Selecting points to display
#     loc = np.where(x_r_m2[:data_size].transpose()[0] == t)[0][:points]
#     x_sample_m2 = []
#     for i in loc:
#         x_sample_m2.append(x_r_m2[:data_size][i][1:])
#     pts = np.asarray(x_sample_m2).transpose()
#     ## Contour
#     (x,y,z) = NN_contour_neo(t, NN_agg_coa_data_m2, c_in_m2, c_out_m2, mcall2, net = 'agg', \
#                              nn_val = 'cont', nn_dt = coa_dt, down = 0.8, up = 1.8, display_time = False)
    
#     time_val_m2.append(z)
#     contours.append(plt.contour(x, y, z, [0], colors='black'))
#     plt.clabel(contours[-1], inline=True, fontsize=10)
#     plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
#                origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
#     plt.colorbar()
#     plt.plot(np.array([80,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
#     plt.plot(np.array([100,100]), np.array([80,100]), linewidth=1, color= 'black', linestyle='dashdot')
#     plt.scatter(pts[0], pts[1], s = 1.5, marker='o', c = 'black')
#     plt.title('Agg NN - Stage 2 - Map: '+str(np.round(t, 4))+' Loop: 0 Price: '+str(np.round(np.mean(r_agg_data_m2),4)))
#     plt.savefig('Max-Call-agg-NN-relu-loop-0-Map-'+str(np.round(t, 4))+'-paths-'+str(data_size)+'-wide.png', dpi=1000)
#     plt.clf()
#     print('Map:', np.round(t,3),'Time:', np.round(time.time()-start_time,2), 'sec')

# ## Stage 3: adding training paths 
    
# for d in range(1, int(len(x_r_m2)/data_size)):
#     x_stage2_m2 = x_r_m2[data_size*d:data_size*(d+1)]
#     y_stage2_m2 = y_r_m2[data_size*d:data_size*(d+1)]
    
#     # Scaling additional data inputs and outputs   
#     input_train_scaled_agg = []
#     for j in range(mcall2['dim']+1):
#         input_train = x_stage2_m2.transpose()[j].reshape(-1, 1)
#         input_train_scaled_agg.append(c_in_m2[j].transform(input_train))
#     x_ = np.stack(input_train_scaled_agg).transpose()[0]
#     y_ = c_out_m2.transform(y_stage2_m2.reshape(-1,1))
    
#     # Neural Network updating parameteres
#     epnum = 5
#     batch_num = 64
#     lossfct = 'mean_squared_error'
#     # optim = 'adam'
#     # optim = tf.keras.optimizers.Adam(learning_rate=0.001)
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
#     NN_agg_coa_data_m2.compile(optimizer = opt, loss = lossfct)
#     NN_agg_coa_data_m2.fit(x_, y_, epochs = epnum, batch_size = batch_num, verbose = 0)
    
    
#     # Computing the option price
#     r_agg_data_m2 = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_data_m2, c_in_m2, \
#                               c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = coa_dt, display_time=False)
        
#     print('Loop:', d, 'Price:', np.round(np.mean(r_agg_data_m2), 4))

#     # Contours of the Aggregate Neural Network in Stage 3
#     model_update(mcall2, dt = 1/3)
#     norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
#     contours = []

#     # t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
#     for t in [1]:
#         start_time = time.time()
#         ## Selecting points to display
#         loc = np.where(x_stage2_m2.transpose()[0] == t)[0][:points]
#         x_sample_m2 = []
#         for i in loc:
#             x_sample_m2.append(x_r_m2[:data_size][i][1:])
#         pts = np.asarray(x_sample_m2).transpose()
#         ## Contour
#         (x,y,z) = NN_contour_neo(t, NN_agg_coa_data_m2, c_in_m2, c_out_m2, mcall2, net = 'agg', \
#                              nn_val = 'cont', nn_dt = coa_dt, down = 0.8, up = 1.8, display_time = False)
        
#         time_val_m2.append(z)
#         contours.append(plt.contour(x, y, z, [0], colors='black'))
#         plt.clabel(contours[-1], inline=True, fontsize=10)
#         plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
#                origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
#         plt.colorbar()
#         plt.plot(np.array([80,100]), np.array([100,100]), linewidth=1, color= 'black', linestyle='dashdot')
#         plt.plot(np.array([100,100]), np.array([80,100]), linewidth=1, color= 'black', linestyle='dashdot')
#         plt.scatter(pts[0], pts[1], s = 1.5, marker='o', c = 'black')
#         plt.title('Agg NN - Stage 3 - Map: '+str(np.round(t, 4))+' Loop: '+str(d)+' Price: '+str(np.round(np.mean(r_agg_data_m2),4)))
#         plt.savefig('Max-Call-agg-NN-relu-loop-'+str(d)+'-Map-'+str(np.round(t, 4))+'-paths-'+str(data_size)+'-wide.png', dpi=1000)
#         plt.clf()
#         print('Map:', np.round(t,3),'Time:', np.round(time.time()-start_time,2), 'sec')


# ## Stability Metrics

# np.abs(np.asarray(time_val_m2)[1] - np.asarray(time_val_m2)[0])
# # Metric -- Squared difference between timing values
# sq_dif_m2 = []
# # Square difference between timing values
# abs_dif_m2 = []
# for x in range(1, int(len(x_r_m2)/data_size)):
#     sq_dif_m2.append(np.sum((np.asarray(time_val_m2)[x] - np.asarray(time_val_m2)[x-1])**2))
#     abs_dif_m2.append(np.sum(np.abs(np.asarray(time_val_m2)[x] - np.asarray(time_val_m2)[x-1])))

# plt.plot(range(1, len(sq_dif_m2)+1), np.asarray(sq_dif_m2)/sq_dif_m2[0])
# plt.plot(range(1, len(sq_dif_m2)+1), np.repeat(1,len(sq_dif_m2)), linewidth=1, color= 'black', linestyle='dashdot')
# plt.xlabel('Level')
# plt.ylabel('Loop')
# plt.title('Max Call - Squared Diff - Paths: '+str(data_size))
# plt.savefig('MCall-Stability-sq-diff-dt-'+str(coa_dt)+'-paths-'+str(data_size)+'-wide.png', dpi=1000)
# plt.clf()

# plt.plot(range(1, len(abs_dif_m2)+1), np.asarray(abs_dif_m2)/abs_dif_m2[0])
# plt.plot(range(1, len(abs_dif_m2)+1), np.repeat(1,len(abs_dif_m2)), linewidth=1, color= 'black', linestyle='dashdot')
# plt.xlabel('Level')
# plt.ylabel('Loop')
# plt.title('Max Call - Absolute Diff - Paths: '+str(data_size))
# plt.savefig('MCall-Stability-abs-diff-dt-'+str(coa_dt)+'-paths-'+str(data_size)+'-wide.png', dpi=1000)
# plt.clf()

# '''
# ### TO BE DELETED -- OLD Version

# # Metric

# np.abs(np.asarray(time_val_m2)[1] - np.asarray(time_val_m2)[0])
# sq_dif = []
# abs_dif = []
# for x in range(1, int(len(x_r_m2)/data_size)):
#     start_time = time.time()
#     sq_dif.append(np.sum((np.asarray(time_val_m2)[x] - np.asarray(time_val_m2)[x-1])**2))
#     abs_dif.append(np.sum(np.abs(np.asarray(time_val_m2)[x] - np.asarray(time_val_m2)[x-1])))
#     print('x:', x,'Time:', np.round(time.time()-start_time,2), 'sec')
    
# (np.asarray(sq_dif)[1:]-np.asarray(sq_dif)[:-1])/np.asarray(sq_dif)[:-1]  

# (np.asarray(sq_dif)[-1] - np.asarray(sq_dif)[0])/np.asarray(sq_dif)[0]

# np.round((np.asarray(abs_dif)[1:] - np.asarray(abs_dif)[:-1])/np.asarray(abs_dif)[:-1],4)

# np.round((np.asarray(abs_dif)[-1] - np.asarray(abs_dif)[0])/np.asarray(abs_dif)[0],4)
# '''

# ##################################
# ### Part of Diagnostics -- End ###
# ##################################


# ## Aggregate Neural Network -- Coarse Time Grid
# epnum = 5
# np.random.seed(16)
# tf.random.set_seed(16)
# # Same Paths
# NN_agg_coa_m2 = NN_aggregate_neo(mcall2, NN_seq_coa_m2, c_in_m2, c_out_m2, nn_val = 'cont', \
#                             stock = stock_coa_m2, node_num = 25, batch_num = 64, \
#                             epoch_num = epnum, display_time=True)
    
# # Training Data
# np.random.seed(16)
# tf.random.set_seed(16)
# NN_agg_coa_data_m2 = NN_aggregate_neo(mcall2, NN_seq_coa_m2, c_in_m2, c_out_m2, nn_val = 'cont', \
#                             data = True, x = x_m2, y = y_m2, node_num = 25, batch_num = 64, \
#                             epoch_num = epnum, display_time=True)
    
# ## Testing 
# # Initializing Fine Test Grid
# model_update(mcall2, dt = 1/3)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m2 = stock_sim(100000, mcall2)

# # [0.2, 0.1, 0.05, 0.025]
# for dt in [0.2, 0.1, 0.05, 0.025]:
#     start_time = time.time()
#     model_update(mcall2, dt = dt)
#     stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, dt) # Coarse Test Grid
#     # Same Paths
#     (r_agg_m2, stop_agg_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_m2, c_in_m2, c_out_m2, \
#                              val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
        
#     # Training Data 
#     (r_agg_data_m2, stop_agg_data_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_data_m2, c_in_m2, c_out_m2, \
#                              val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
     
#     print('dt:', dt)
#     # Same Paths  
#     print('Price - agg NN - Path:', np.round(np.mean(r_agg_m2), 4))
#     print('Avg Std agg NN - Path:', np.round(np.std(r_agg_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))
#     # Training Data 
#     print('Price - agg NN - Data:', np.round(np.mean(r_agg_data_m2), 4))
#     print('Avg Std agg NN - Data:', np.round(np.std(r_agg_data_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))
#     print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,2), 'sec')
    
    
# ### Monitoring differences
# np.amax(stop_agg_data_m2 - stop_agg_m2)
# np.where((stop_agg_data_m2 - stop_agg_m2) == np.amax(stop_agg_data_m2 - stop_agg_m2))[0][:20]
# np.amin(stop_agg_data_m2 - stop_agg_m2)
# np.where((stop_agg_data_m2 - stop_agg_m2) == np.amin(stop_agg_data_m2 - stop_agg_m2))[0]
# ## Max differences 
# # Paths 168, 298, 310, 511,  558,  563,  593,  728,  765,  779
# # Min differences
# ## Paths 33781, 56589
# # Interesting paths
# # 27007, 77474

# ## Paths
# path = 74820
# model_update(mcall2, dt = 0.025)
# t_steps = np.arange(mcall2['dt'], mcall2['T']+mcall2['dt'], mcall2['dt'])
# plt.plot(t_steps, stockFwd_fine_m2[:, path])
# plt.axhline(y = 100, color='black', linestyle='dashed', linewidth=1)
# plt.axvline(x = stop_agg_m2[path], color = 'b', label = 'path', \
#             linestyle='dashed', linewidth=1)
# plt.axhline(y = mcall2['K']+r_agg_m2[path]*np.exp(mcall2['r']*stop_agg_m2[path]), color = 'b', label = 'data', \
#             linestyle='dashed', linewidth=1)
# plt.axvline(x = stop_agg_data_m2[path], color = 'm', label = 'data', \
#             linestyle='dashed', linewidth=1)
# plt.axhline(y = mcall2['K']+r_agg_data_m2[path]*np.exp(mcall2['r']*stop_agg_data_m2[path]), \
#             color = 'm', label = 'data', linestyle='dashed', linewidth=1)
# plt.xlabel('Time')
# plt.ylabel('Stock Value')
# plt.title('Path: '+str(path)+' Stop Path: '+str(stop_agg_m2[path])+' Stop Data: '\
#           +str(stop_agg_data_m2[path]))
# plt.savefig('Loop-1-Path-'+str(path)+'-stop-data-'+str(np.round(stop_agg_data_m2[path],3))+\
#             '-stop-path-'+str(np.round(stop_agg_m2[path],3))+'.png', dpi=1000)

# plt.scatter(stop_agg_m2[:500], stop_agg_data_m2[:500], s=1, c='black', marker='o')
# plt.plot([0, 1], [0, 1], color = 'red', linewidth=0.5)
# plt.xlabel('Path')
# plt.ylabel('Data')
    
# t_inc = 0.01
# p = []
# for x in range(160,190,5):
#     for y in range(190,195,5):
#         pos = np.array([x, y])
#         (x_data, y_data) = NN_time_neo(pos, NNagg, c_in_m2, c_out_m2, mcall2, \
#                                nn_val = 'cont', nn_dt = coa_dt, t_inc=t_inc, display_time = False)
#         p.append(list(pos))
#         plt.plot(x_data, y_data, linewidth=1)
# plt.plot(x_data, np.repeat(0, 100), color='black', linestyle='dashed', linewidth=1)
# plt.legend(p)
# plt.xlabel('Time')
# plt.ylabel('Timing Value')
# plt.title('Timing Values dt: '+str(t_inc)+' Loop 2 NN: '+str(0.025))
# plt.savefig('Timing-values-dt-'+str(coa_dt)+'-pos-list-160-190-at-100-opposite.png', dpi=1000)
        

# ## Testing 
# # Initializing Fine Test Grid
# model_update(mcall2, dt = 0.025)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m2 = stock_sim(100000, mcall2)

# # [0.2, 0.1, 0.05, 0.025]
# for dt in [0.2, 0.1, 0.05, 0.025]:
#     start_time = time.time()
#     model_update(mcall2, dt = dt)
#     stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, dt) # Coarse Test Grid
#     # Same Paths
#     (r_new_m2, stop_new_m2) = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NNagg, c_in_m2, c_out_m2, \
#                              val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
         
#     print('dt:', dt)
#     # Same Paths  
#     print('Price - agg NN - New:', np.round(np.mean(r_new_m2), 4))
#     print('Avg Std agg NN - New:', np.round(np.std(r_new_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))
#     print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,2), 'sec')
    

# ### Grid debugging
# ## Paths
# path = 75
# model_update(mcall2, dt = 0.025)
# t_steps = np.arange(mcall2['dt'], mcall2['T']+mcall2['dt'], mcall2['dt'])
# plt.plot(t_steps, stock[:, path])
# plt.axhline(y = 100, color='black', linestyle='dashed', linewidth=1)
# plt.axvline(x = stop_val[path], color = 'b', label = 'path', \
#             linestyle='dashed', linewidth=1)
# plt.axhline(y = mcall2['K']+q_val[path]*np.exp(mcall2['r']*stop_val[path]), color = 'b', \
#             label = 'data', linestyle='dashed', linewidth=1)
# plt.axvline(x = stop_o[path], color = 'm', label = 'data', \
#             linestyle='dashed', linewidth=1)
# plt.axhline(y = mcall2['K']+q_o[path]*np.exp(mcall2['r']*stop_o[path]), \
#             color = 'm', label = 'data', linestyle='dashed', linewidth=1)
# plt.xlabel('Time')
# plt.ylabel('Stock Value')
# plt.title('Grid Path: '+str(path)+' Stop Loop: '+str(stop_val[path])+' Stop Regular: '\
#           +str(stop_o[path]))
# plt.savefig('Grid-Path-'+str(path)+'-stop-loop-'+str(np.round(stop_val[path],3))+\
#             '-stop-reg-'+str(np.round(stop_o[path],3))+'.png', dpi=1000)

    
# np.amin(q_val - q_o)
# np.amax(stop_val - stop_o)
# np.where((stop_val - stop_o) == np.amax(stop_val - stop_o))
# np.where((q_val - q_o) == np.amax(q_val - q_o))
# np.where((q_val - q_o) == np.amin(q_val - q_o))

# ###########################
# ### Stage 3 -- Updating ###
# ###########################

# ### Setting up the loops and testing framework 
# ## Testing 
# # Initializing Fine Test Grid
# model_update(mcall2, dt = 0.025)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m2 = stock_sim(100000, mcall2)
# nn_dt = 0.2
# coa_dt = 0.05
# model_update(mcall2, dt = coa_dt)
# stockFwd_coa_m2 = stock_thin(stockFwd_fine_m2, mcall2, coa_dt)

# # Training Data 
# np.random.seed(25)
# tf.random.set_seed(25)
# NN_loop_v5(1, 8, mcall2, NN_agg_coa_data_m2, c_in_m2, c_out_m2, 'cont', nn_dt, 0, \
#            stockFwd_coa_m2, epoch_num = 5, factor = 0.7, display_time=True)

# # Same Paths
# np.random.seed(25)
# tf.random.set_seed(25)
# NN_loop_v5(1, 8, mcall2, NN_agg_coa_m2, c_in_m2, c_out_m2, 'cont', nn_dt, 0, \
#            stockFwd_coa_m2, epoch_num = 5, factor = 0.7, display_time=True)

# # Training Data 
# r_agg_data_m2 = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_data_m2, c_in_m2, \
#                               c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = nn_dt, display_time=False)
    
# # Same Paths
# r_agg_m2 = NN_payoff_neo(0, stockFwd_coa_m2, mcall2, 'agg', NN_agg_coa_m2, c_in_m2, \
#                          c_out_m2, val = 'cont', nn_val = 'cont', nn_dt = nn_dt, display_time=False)   
    
# print('Price - updated agg NN - Data:', np.round(np.mean(r_agg_data_m2), 4))
# print('Price - updated agg NN - Path:', np.round(np.mean(r_agg_m2), 4))

# model_update(mcall2, dt = 0.2)
# # Aggregate Neural Network 
# # Same Paths  
# norm_max = matplotlib.colors.Normalize(vmin=-15, vmax=15)
# contours = []
# contdata1 = []

# # t_steps = np.arange(mcall2['dt'], mcall2['T'], mcall2['dt'])
# for t in [0.4, 0.9]:
#     start_time = time.time()
#     (x,y,z) = NN_contour_neo(t, NN_agg_coa_data_m2, c_in_m2, c_out_m2, mcall2, net = 'agg', \
#                              nn_val = 'cont', nn_dt = nn_dt, display_time = True)
    
#     contdata1.append((x,y,z))
#     contours.append(plt.contour(x, y, z, [0], colors='black'))
#     plt.clabel(contours[-1], inline=True, fontsize=10)
#     plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
#                origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
#     plt.colorbar()
#     plt.title('Agg NN - Data - Map: '+str(np.round(t, 4))+' Loop: 1 Price: '+str(np.round(np.mean(r_agg_data_m2),4)))
#     plt.savefig('Max-Call-agg-NN-loop-1-data-Map-'+str(np.round(t, 4))+'-size-64.png', dpi=1000)
#     plt.clf()
#     print('Map:',np.round(t,3),'Time:', np.round(time.time()-start_time,2), 'sec')
   
# # Diff
# norm_max = matplotlib.colors.Normalize(vmin=-17, vmax=17)
# contours = []
# t = [0.4, 0.9]

# for i in range(2):
#     start_time = time.time()
#     x = contdata0[i][0]
#     y = contdata0[i][1]
#     z = contdata1[i][2] - contdata[i][2]
#     contours.append(plt.contour(x, y, z, [0], colors='black'))
#     plt.clabel(contours[-1], inline=True, fontsize=10)
#     plt.imshow(np.array(z, dtype = float), extent=[80, 180, 80, 180], \
#                origin='lower', cmap='Spectral', norm = norm_max, alpha=1)
#     plt.colorbar()
#     plt.title('Agg NN - Path - Map: '+str(np.round(t[i], 4))+' Diff - Loop: 1 Price: '+str(np.round(np.mean(r_agg_m2),4)))
#     plt.savefig('Max-Call-agg-NN-diff-loop-1-path-Map-'+str(np.round(t[i], 4))+'.png', dpi=1000)
#     plt.clf()
#     print('Map:',np.round(t[i],3),'Time:', np.round(time.time()-start_time,2), 'sec')    
   
# ## Plot Paths
# # Paths 1, 99997
# # Max Paths 27007, 33781, 56589, 77474
# # Min Paths 779,  2749,  2864,  4183,  4769,  5008,  9441,  9682 among others
# path = 9441
# model_update(mcall2, dt = 0.025)
# t_steps = np.arange(mcall2['dt'], mcall2['T']+mcall2['dt'], mcall2['dt'])
# plt.plot(t_steps, stockFwd_fine_m2[:, path])
# plt.axhline(y = 100, color='black', linestyle='dashed', linewidth=1)
# plt.axvline(x = stop_loop_m2[path], color = 'black', label = 'Loop', \
#             linestyle='dashed', linewidth=1)
# plt.axvline(x = stop_agg_m2[path], color = 'black', label = 'Loop', \
#             linestyle='dashed', linewidth=1)
# plt.xlabel('Time')
# plt.ylabel('Stock Value')
# plt.title('Path: '+str(path)+' Stop NN: '+str(stop_agg_m2[path])+' Stop loop: '\
#           +str(stop_loop_m2[path]))
# plt.savefig('Path-'+str(path)+'.png', dpi=1000)



# # Same Paths
# r_agg_u_m2 = NN_payoff_neo(0, stockFwd_coa_m2[:,:10000], NN_agg_coa_m2, c_in_m2, \
#                                 c_out_m2, mcall2, net = 'agg', display_time=True)

# print('Price with Agg NN - Paths:', np.round(np.mean(r_agg_u_m2), 4))
# print('Avg Std Agg NN    - Paths:', np.round(np.std(r_agg_u_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))
# print('Price with Agg NN - Data:', np.round(np.mean(r_agg_data_u_m2), 4))
# print('Avg Std Agg NN    - Data:', np.round(np.std(r_agg_data_u_m2, ddof = 1)/np.sqrt(len(stockFwd_coa_m2[0])), 4))


# plt.scatter(r_agg_data_u_m2[:5000], r_loop_data_m2[:5000], s=1, c='black', marker='o')
# plt.plot([-0.2, 10], [-0.2, 10], color = 'red', linewidth=0.5)
# plt.xlabel('Aggregate')
# plt.ylabel('Loop')

# plt.scatter(r_agg_m2[:200], r_loop_m2[:200], s=1, c='black', marker='o')
# plt.plot([-0.2, 100], [-0.2, 100], color = 'red', linewidth=0.5)
# plt.xlabel('Aggregate')
# plt.ylabel('Loop')
# plt.title('Agg NN - Paths :'+str(np.round(np.mean(r_agg_m2),4))+'1 Loop Price: '+str(np.round(np.mean(r_loop_m2),4)))

# #############
# ### OTHER ###
# #############

# path = 19
# model_update(mcall2, dt = 0.05)
# t_steps = np.arange(0.5, mcall2['T'], mcall2['dt'])
# plt.plot(t_steps, stock[:, path])
# plt.xlabel('Time')
# plt.ylabel('Stock Value')
# plt.title('Path: '+str(path))

# #######################
# #### 5D - Max Call ####
# #######################

# mcall5 = {'dim': 5, 'K': 100, 'x0': np.repeat(70, 5), 
#          'sigma': np.array([0.08,0.16,0.24,0.32,0.4]), 
#          'r': 0.05, 'div': np.repeat(0.1, 5), 'T': 3, 'dt': 1/3, 
#          'payoff.func': 'maxi.call.payoff'}

# # Initializing -- Fine Time Grid
# model_update(mcall5, dt = 1/3)
# np.random.seed(15)
# tf.random.set_seed(15)
# stock_fine_m5 = stock_sim(50000, mcall5)
# (c_in_m5, c_out_m5) = scaler(stock_fine_m5, mcall5)

# # Thin at coarse dt
# coa_dt = 1/3
# model_update(mcall5, dt = coa_dt)
# stock_coa_m5 = stock_thin(stock_fine_m5, mcall5, coa_dt) # Coarse time grid

# ## Sequence of Neural Networks -- Coarse Time Grid
# epnum = 5
# np.random.seed(15)
# tf.random.set_seed(15)
# (NN_seq_coa_m5, x_m5, y_m5) = NN_seq_train_neo(stock_coa_m5, mcall5, c_in_m5, c_out_m5, \
#                                     theta = 'average', data = True, val = 'cont', \
#                                     node_num = 25, epoch_num = epnum, display_time=True)
    
# ########################################
# ### Part of Diagnostics -- Beginning ###    
# ########################################

# # Shuffling data
# merge = np.append(x_m5.transpose(), y_m5.transpose()).reshape(mcall5['dim']+2, len(y_m5)).transpose()
# np.random.shuffle(merge)
# x_r_m5 = merge.transpose()[:-1].transpose()
# y_r_m5 = merge.transpose()[-1].reshape(-1, 1)
    
# # Split data
# # split_point = int(np.round(len(x_r_m5)*0.99))
# split_point = len(x_r_m5)-1000
# x_stage2_m5 = x_r_m5[:split_point]
# y_stage2_m5 = y_r_m5[:split_point]
# x_stage3_m5 = x_r_m5[split_point:]
# y_stage3_m5 = y_r_m5[split_point:]

# # Training the Aggregate network on the stage 2 data 
# np.random.seed(16)
# tf.random.set_seed(16)
# NN_agg_coa_data_m5 = NN_aggregate_neo(mcall5, NN_seq_coa_m5, c_in_m5, c_out_m5, nn_val = 'cont', \
#                         data = True, x = x_stage2_m5, y = y_stage2_m5, node_num = 25, batch_num = 64, \
#                         epoch_num = epnum, display_time=True)

# ### Setting up the loops and testing framework 
# ## Testing 
# # Initializing Fine Test Grid
# model_update(mcall5, dt = 1/3)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m5 = stock_sim(100000, mcall5)
# nn_dt = 1/3
# coa_dt = 1/3
# model_update(mcall5, dt = coa_dt)
# stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall5, coa_dt)

# # Computing the option price
# r_agg_data_m5 = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_data_m5, c_in_m5, \
#                               c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = nn_dt, display_time=False)
        
# print('Price - agg NN - Stage 2:', np.round(np.mean(r_agg_data_m5), 4))

# ## Add additional paths -- coresponding to a loop in Stage 3

# # Scaling additional data inputs and outputs   
# input_train_scaled_agg = []
# for j in range(mcall5['dim']+1):
#     input_train = x_stage3_m5.transpose()[j].reshape(-1, 1)
#     input_train_scaled_agg.append(c_in_m5[j].transform(input_train))
# x_ = np.stack(input_train_scaled_agg).transpose()[0]
# y_ = c_out_m5.transform(y_stage3_m5.reshape(-1,1))

# epnum = 5
# batch_num = 64
# NN_agg_coa_data_m5.fit(x_, y_, epochs = epnum, batch_size = batch_num, verbose = 1)

# # Computing the option price
# r_agg_data_m5 = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_data_m5, c_in_m5, \
#                               c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = nn_dt, display_time=False)
        
# print('Price - agg NN - Stage 3:', np.round(np.mean(r_agg_data_m5), 4))

# ##################################
# ### Part of Diagnostics -- End ###
# ##################################    

# ## Aggregate Neural Network -- Coarse Time Grid
# epnum = 5
# np.random.seed(16)
# tf.random.set_seed(16)
# # Same Paths
# NN_agg_coa_m5 = NN_aggregate_neo(mcall5, NN_seq_coa_m5, c_in_m5, c_out_m5, nn_val = 'cont', \
#                             stock = stock_coa_m5, node_num = 25, batch_num = 64, \
#                             epoch_num = epnum, display_time=True)
    
# # Training Data
# np.random.seed(16)
# tf.random.set_seed(16)
# NN_agg_coa_data_m5 = NN_aggregate_neo(mcall5, NN_seq_coa_m5, c_in_m5, c_out_m5, nn_val = 'cont', \
#                             data = True, x = x_m5, y = y_m5, node_num = 25, batch_num = 64, \
#                             epoch_num = epnum, display_time=True)
    
# ## Testing 
# # Initializing Fine Test Grid
# model_update(mcall5, dt = 1/3)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m5 = stock_sim(100000, mcall5)

# for dt in [1/3]:
#     start_time = time.time()
#     model_update(mcall5, dt = dt)
#     stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall5, dt) # Coarse Test Grid
#     # Same Paths
#     (r_agg_m5, stop_agg_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_m5, c_in_m5, c_out_m5, \
#                              val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
        
#     # Training Data 
#     (r_agg_data_m5, stop_agg_data_m5) = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_data_m5, c_in_m5, c_out_m5, \
#                              val = 'cont', nn_val = 'cont', nn_dt = coa_dt, stop=True, display_time=False)
     
#     print('dt:', dt)
#     # Same Paths  
#     print('Price - agg NN - Path:', np.round(np.mean(r_agg_m5), 4))
#     print('Avg Std agg NN - Path:', np.round(np.std(r_agg_m5, ddof = 1)/np.sqrt(len(stockFwd_coa_m5[0])), 4))
#     # Training Data 
#     print('Price - agg NN - Data:', np.round(np.mean(r_agg_data_m5), 4))
#     print('Avg Std agg NN - Data:', np.round(np.std(r_agg_data_m5, ddof = 1)/np.sqrt(len(stockFwd_coa_m5[0])), 4))
#     print('dt: '+str(dt)+' Time:', np.round(time.time()-start_time,5), 'sec')
    

# ###########################
# ### Stage 3 -- Updating ###
# ###########################

# ### Setting up the loops and testing framework 
# ## Testing 
# # Initializing Fine Test Grid
# model_update(mcall5, dt = 1/6)
# np.random.seed(18)
# tf.random.set_seed(18)
# stockFwd_fine_m5 = stock_sim(100000, mcall5)
# nn_dt = 1/3
# coa_dt = 1/3
# model_update(mcall5, dt = coa_dt)
# stockFwd_coa_m5 = stock_thin(stockFwd_fine_m5, mcall5, coa_dt)

# # Training Data 
# np.random.seed(25)
# tf.random.set_seed(25)
# NN_loop_v5(5, 16, mcall5, NN_agg_coa_data_m5, c_in_m5, c_out_m5, 'cont', nn_dt, 1, \
#            stockFwd_coa_m5, epoch_num = 5, factor = 0.5, display_time=True)

# # Same Paths
# np.random.seed(25)
# tf.random.set_seed(25)
# NN_loop_v5(10, 8, mcall5, NN_agg_coa_m5, c_in_m5, c_out_m5, 'cont', nn_dt, 1, \
#            stockFwd_coa_m5, epoch_num = 5, factor = 0.5, display_time=True)

# # Training Data 
# r_agg_data_m5 = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_data_m5, c_in_m5, \
#                               c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = nn_dt, display_time=False)
    
# # Same Paths
# r_agg_m5 = NN_payoff_neo(0, stockFwd_coa_m5, mcall5, 'agg', NN_agg_coa_m5, c_in_m5, \
#                          c_out_m5, val = 'cont', nn_val = 'cont', nn_dt = nn_dt, display_time=False)   
    
# print('Price - updated agg NN - Data:', np.round(np.mean(r_agg_data_m5), 4))
# print('Price - updated agg NN - Path:', np.round(np.mean(r_agg_m5), 4))
    