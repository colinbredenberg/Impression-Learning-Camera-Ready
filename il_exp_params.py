import os
import numpy as np
mode = 'standard'
local = True
local_plot = True
save = False
layered = True
if local == False:
    array_num = int(os.environ['SLURM_ARRAY_TASK_ID']);
else:
    array_num = 1#8

alg_num = 1
#%% Parameters for normal network training
if mode == 'standard':
    mode = 'standard'
    algorithm = 'wake_sleep'
    recognition_scale = 1
    #n_latent = 20 #number of latent dimensions in the data
    #n_out = 100 #number of output dimensions in the data
    #n_in = 100 #number of input dimensions for neurons
    n_latent = 20
    n_out = 100
    n_in = 100
    n_neurons = 20 #number of latent dimensions for neurons
    n_sample = 2000000 #number of data points for the train dataset
    n_test = 30000 #number of data points for the test dataset
    dt = 0.1 #time step for the data OU process
    
    sigma_latent = 0.01 #latent noise for the network
    sigma_latent_gen = 0.01
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    epoch_num = 1
    learning_rate = 1e-3#1e-7 #learning rate
    switch_period = 1#int(n_sample/600000) #number of samples taken before switching from wake to sleep
elif mode == 'time_constant':
    algorithm = 'wake_sleep'
    recognition_scale = 1;
    n_latent = 4 #number of latent dimensions in the data
    n_out = 10 #number of output dimensions in the data
    n_in = 10 #number of input dimensions for neurons
    n_neurons = 4 #number of latent dimensions for neurons
    n_sample = 2000000 #number of data points for the train dataset
    n_test = 5000 #number of data points for the test dataset
    dt_list = np.log10(np.logspace(1, 0.01, num = 10))
    dt = dt_list[array_num - 1] #time step for the data OU process
    sigma_latent = 0.01 #latent noise for the network
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    learning_rate = 5e-4 #learning rate
    switch_period = 1#int(n_sample/600000) #number of samples taken before switching from wake to sleep
elif mode == 'switch_period':
    algorithm = 'wake_sleep'
    recognition_scale = 1;
    n_latent = 4 #number of latent dimensions in the data
    n_out = 10 #number of output dimensions in the data
    n_in = 10 #number of input dimensions for neurons
    n_neurons = 4 #number of latent dimensions for neurons
    n_sample = 1000000 #number of data points for the train dataset
    n_test = 30000 #number of data points for the test dataset
    dt = 0.1 #time step for the data OU process
    epoch_num = 1
    sigma_latent = 0.01 #latent noise for the network
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    learning_rate = 5e-4 #learning rate
    switch_period_list = np.arange(1,33,3)#np.arange(1,11,1)
    switch_period = switch_period_list[array_num -1]#int(n_sample/600000) #number of samples taken before switching from wake to sleep
#%% Parameters for comparing the SNR across different algorithms across learning
elif mode == 'SNR':
    algorithm = 'wake_sleep'
    recognition_scale = 1;
    n_latent = 2 #number of latent dimensions in the data
    n_out = 4 #number of output dimensions in the data
    n_in = 4 #number of input dimensions for neurons
    n_neurons = 2 #number of latent dimensions for neurons
    n_sample = 600000 #number of data points for the train dataset
    n_test = 10000#1000000 #number of data points for the test dataset
    n_compare = 4
    dt = 0.1 #time step for the data OU process
    
    sigma_latent = 0.01 #latent noise for the network
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    learning_rate = 1e-4 #learning rate
    switch_period = 1#int(n_sample/600000) #number of samples taken before switching from wake to sleep
    epoch_num_snr = 1000000

#%% Parameters for seeing how dimensionality affects SNR
elif mode == 'dimensionality':
    if array_num <= 5:
        algorithm = 'wake_sleep'
        #recognition_scale = 1
        #recognition_scale = 3
        learning_rate = 1e-2 / (10**((np.mod(1 - 1, 20)+1)/2)) #learning rate set by lr_optim
    elif array_num <= 10:
        algorithm = 'reinforce'
        #recognition_scale = 17
        #recognition_scale = 2
        learning_rate = 1e-2 / (10**((np.mod(35 - 1, 20)+1)/2)) #learning rate set by lr_optim
    
    recognition_scale = 1
    #learning_rate = 1e-3
    n_latent = 2**(np.mod(array_num - 1, 5)+1)#10 * (np.mod(array_num - 1, 10)+1) #number of latent dimensions in the data
    n_out = 2*2**(np.mod(array_num - 1, 5)+1)#100 #number of output dimensions in the data
    n_in = 2*2**(np.mod(array_num - 1, 5)+1)#100 #number of input dimensions for neurons
    n_neurons = 2**(np.mod(array_num - 1, 5)+1)#10 * (np.mod(array_num - 1, 10)+1) #number of latent dimensions for neurons
    n_sample = 3600000 #number of data points for the train dataset
    n_test = 30000 #number of data points for the test dataset
    dt = 0.1 #time step for the data OU process
    
    epoch_num = 1
    
    sigma_latent = 0.01 #latent noise for the network
    sigma_latent_gen = 0.01
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    
    switch_period = 1#int(n_sample/600000) #number of samples taken before switching from wake to sleep
elif mode == 'lr_optim':
    if array_num <= 20:
        algorithm = 'wake_sleep'
    elif array_num <= 40:
        algorithm = 'reinforce'
    n_latent = 2 #number of latent dimensions in the data
    n_out = 4 #number of output dimensions in the data
    n_in = 4 #number of input dimensions for neurons
    n_neurons = 2 #number of latent dimensions for neurons
    n_sample = 3600000 #number of data points for the train dataset
    n_test = 30000 #number of data points for the test dataset
    dt = 0.1 #time step for the data OU process
    epoch_num = 1
    
    sigma_latent = 0.01 #latent noise for the network
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    
    learning_rate = 1e-2 / (10**((np.mod(array_num - 1, 20)+1)/2))
    recognition_scale = 1#(np.mod(array_num - 1, 10)+1) #learning rate
    switch_period = 1#int(n_sample/600000) #number of samples taken before switching from wake to sleep

    
    
elif mode == 'MNIST':
    if array_num == 1:
        algorithm = 'wake_sleep'
        recognition_scale = 1
    elif array_num == 2:
        algorithm = 'backprop'
        recognition_scale = 1
    elif array_num >= 3:
        algorithm = 'reinforce'
        lr_exp = (array_num - 3)
        recognition_scale = 10**lr_exp #1 * 10**(-1 * lr_exp)
    learning_rate = 1e-3
    n_latent = 40
    dt = 0.1
    n_out = 28**2
    n_in = n_out
    n_neurons = 100
    #n_neurons = 40
    repeat_length = 10 #how many time steps to show a given image
    n_sample = 50000 #number of training data points
    n_digits = 10 #number of digits to extract from the MNIST data set
    n_test = 100 #number of testing data points
    #sigma_latent = 0.01 #latent noise for the network
    sigma_latent = 0.01#np.sqrt(0.1)
    sigma_latent_gen = 0.5#0.05#np.sqrt(0.1)#0.05#np.sqrt(0.1)
    sigma_obs_gen = 0.05 #observation noise for the network
    sigma_in = 0.01
    
    epoch_num = 3
    gen_sim_num = 20
    
    #learning_rate = 1e-3 #learning rate
    switch_period = 1 #number of time steps to wait for switching from wake phase to sleep phase
    
elif mode == 'Vocal_Digits':
    if array_num == 1:
        algorithm = 'wake_sleep'
        learning_rate = 5e-4
    elif array_num == 2:
        algorithm = 'backprop'
        learning_rate = 1e-3
    elif array_num >= 3:
        algorithm = 'reinforce'
        lr_exp = (array_num - 3)
        learning_rate = 1e-3 * 10**(-lr_exp)
    recognition_scale = 1#10**lr_exp #1 * 10**(-1 * lr_exp)
    n_latent = 40
    dt = 0.1
    n_out = 128
    n_in = n_out
    n_neurons = 100
    #n_neurons = 40
    n_sample = 0 #number of training data points
    n_digits = 10 #number of digits to extract from the MNIST data set
    n_test = 0 #number of testing data points
    #sigma_latent = 0.01 #latent noise for the network
    sigma_latent = 0.01#np.sqrt(0.1)
    sigma_latent_gen = 0.13#0.05#np.sqrt(0.1)#0.05#np.sqrt(0.1)
    sigma_obs_gen = 0.01 #observation noise for the network
    sigma_in = 0.01
    
    epoch_num = 20
    gen_sim_num = 20
    
    #learning_rate = 1e-3 #learning rate
    switch_period = 1 #number of time steps to wait for switching from wake phase to sleep phase
    
    
    
    