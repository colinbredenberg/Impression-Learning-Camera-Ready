import os
import numpy as np
mode = 'dimensionality'
local = True
local_plot = True
save = False
layered = True
if local == False:
    array_num = int(os.environ['SLURM_ARRAY_TASK_ID']);
else:
    array_num = 1

alg_num = 1

if mode == 'dimensionality':
    if array_num <= 5:
        algorithm = 'backprop'
        #recognition_scale = 1
        #recognition_scale = 3
        learning_rate = 1e-3
    
    n_latent = 2**(np.mod(array_num - 1, 5)+1)#10 * (np.mod(array_num - 1, 10)+1) #number of latent dimensions in the data
    n_out = 2*2**(np.mod(array_num - 1, 5)+1)#100 #number of output dimensions in the data
    n_in = 2*2**(np.mod(array_num - 1, 5)+1)#100 #number of input dimensions for neurons
    n_neurons = 2**(np.mod(array_num - 1, 5)+1)#10 * (np.mod(array_num - 1, 10)+1) #number of latent dimensions for neurons
    n_sample = 300000 #number of data points for the train dataset
    n_test = 30000 #number of data points for the test dataset
    dt = 0.1 #time step for the data OU process
    
    sigma_latent = 0.01 #latent noise for the network
    sigma_latent_gen = 0.01
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    
    switch_period = n_sample #1#int(n_sample/600000) #number of samples taken before switching from wake to sleep