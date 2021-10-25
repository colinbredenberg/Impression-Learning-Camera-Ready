import il_exp_params as exp_params
import numpy as np
import matplotlib.pyplot as plt
from impression_learning import *
import os
import pickle
import statsmodels.api as sm
from sklearn.decomposition import PCA
from matplotlib import rc
from scipy import stats

if exp_params.mode == 'Vocal_Digits':
    import librosa
    import librosa.display

#%% Functions for loading data
def load_file(path):
    data = pickle.load(open(path, 'rb'))
    return data

def unpack_loaded_data(data):
    globals().update(data) #this turns all of the keys in the datafile into variable names
    
local_plot = exp_params.local_plot
load_folder = ''
plot_save_folder = ''
image_save = False
if (not(local_plot)):
    if exp_params.mode == 'standard':
        loss_aggregate = []
        for ii in range(1, 21):
            filename = 'impression_' + str(ii)
            path = os.getcwd() + load_folder + filename
            data = load_file(path)
            unpack_loaded_data(data)
            loss_aggregate.append(loss_mean)
        filename = 'impression_1'
    elif exp_params.mode == 'MNIST':
        filename = 'impression_mnist_1'
    else:
        filename = 'impression_snr_1'
    path = os.getcwd() + load_folder + filename
    data = load_file(path)
    unpack_loaded_data(data)
    if exp_params.mode == 'standard':
        network = test_sim.nn
        latent_test = test_sim.latent
        latent_gen = gen_sim.latent

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
rc('font',**{'family':'serif','serif':['Times']})
n_latent = exp_params.n_latent
n_out = exp_params.n_out
n_in = exp_params.n_in
n_neurons = exp_params.n_neurons
n_sample = exp_params.n_sample
n_test = exp_params.n_test
dt = exp_params.dt
sigma_latent_data = 1 * np.sqrt(dt)
mixing_matrix = np.random.normal(loc = 0, scale = 1/n_latent, size = (n_in, n_latent)) #observation matrix
transition_matrix = (1 - sigma_latent_data**2) * np.eye(n_latent)

#build the neural network

sigma_latent = exp_params.sigma_latent
sigma_obs_gen = exp_params.sigma_obs_gen
sigma_latent_gen = sigma_latent_data

#build the learning algorithm
learning_rate = exp_params.learning_rate
switch_period = exp_params.switch_period
#%% Plotting functions for the standard mode
#Plot the loss through training

if exp_params.mode == 'standard':
    if exp_params.layered:
        W_out = network.l0.W_out
        transition_mat = network.l1.transition_mat
    else:
        W_out = network.W_out
        transition_mat = network.transition_mat
    latent_test = wake_sequence.latent
    fig_dim = (5,5)
    loss_fig = plt.figure(figsize = fig_dim)
    if not(local_plot):
        loss_array = np.array(loss_aggregate)
        loss_array_mean = np.mean(loss_array, axis = 0)
        loss_array_sem = stats.sem(loss_array, axis = 0)
        plt.errorbar(range(0, len(loss_array)), loss_array_mean, yerr = loss_array_sem)
        plt.title('loss through training')
    else:
        plt.plot(loss_mean)
        plt.title('loss through training')
    
    #%%
    #Plot the input and targets
    prediction = np.ndarray.flatten((W_out @ latent_test)[1,0:200:1])
    true = np.ndarray.flatten(data_test[1,0:200:1])
    io_trace_fig = plt.figure(figsize = fig_dim)
    plt.plot(prediction)
    plt.plot(true)
    plt.ylim((-0.5,0.5))
    plt.xlabel('time')
    plt.legend(('prediction', 'true'))
    plt.title('input-output comparison')
    #%%
    io_fig = plt.figure(figsize = fig_dim)
    prediction = np.ndarray.flatten((W_out @ latent_test)[0::5,0:1000:19])
    true = np.ndarray.flatten(data_test[0::5,0:1000:19])
    plt.scatter(true, prediction)
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.plot([-2,2],[-2,2], 'k')
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    #%%
    idx = 0
    acf_fig, ax = plt.subplots(figsize = fig_dim)
    sm.graphics.tsa.plot_acf(latent_test[idx,:], ax = ax, lags=80)
    sm.graphics.tsa.plot_acf(data_latent_test[idx,:], ax = ax, lags = 80)
    sm.graphics.tsa.plot_acf(latent_gen[idx,:], ax = ax, lags = 80)
    plt.legend(('','latent', '', 'stim latent', '', 'gen latent'))
    print(latent_test[0,:].shape)
    #%%

    #compare normal perception to wake-sleep alternating perception
    plt.figure(figsize = fig_dim)
    plt.plot(wake_sleep_sequence.latent[1, 0:400])
    plt.plot(wake_sequence.latent[1,0:400])
    plt.legend(('learning alternation', 'wake phase'))
    
    if image_save:
        loss_fig.savefig(plot_save_folder+'loss.pdf', format = 'pdf')
        io_fig.savefig(plot_save_folder+'io_plot.pdf', format = 'pdf')
        acf_fig.savefig(plot_save_folder+'acf_plot.pdf', format = 'pdf')
        io_trace_fig.savefig(plot_save_folder + 'io_trace.pdf', format = 'pdf')

#%% Plotting functions for the SNR mode
if exp_params.mode == 'SNR':
    fig_dim = (5,5)
    loss_mean_aggregate = []
    corr = np.zeros((11,))
    n_sim = 20
    array_len = len(mean_ws)
    similarity_ws_bp = np.zeros((n_sim, array_len))
    cosine_similarity_ws_r = np.zeros((n_sim, array_len))
    similarity_ws_r = np.zeros((n_sim, array_len))
    gradient_norm_ws = np.zeros((n_sim, array_len))
    gradient_norm_reinforce = np.zeros((n_sim, array_len))
    mean_snr_ws = np.zeros((n_sim, array_len))
    mean_snr_r = np.zeros((n_sim, array_len))
    for jj in range(1, n_sim + 1):
        filename = 'impression_snr_' + str(jj)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_aggregate.append(loss_mean)
        

        for ii in range(0, array_len):
            similarity_ws_r[jj-1, ii] = unnormalized_similarity(mean_ws[ii][1][0], mean_reinforce[ii][1][0])
            cosine_similarity_ws_r[jj-1, ii] = cosine_similarity(mean_ws[ii][1][0], mean_reinforce[ii][1][0])
            gradient_norm_ws[jj-1, ii] = np.linalg.norm(mean_ws[ii][1][0])
            gradient_norm_reinforce[jj-1,ii] = np.linalg.norm(mean_reinforce[ii][1][0])
            mean_snr_ws[jj-1, ii] = np.mean(snr_ws[ii][1][0])
            mean_snr_r[jj-1, ii] = np.mean(snr_reinforce[ii][1][0])

    loss_mean_array = np.array(loss_mean_aggregate)
    fig_dim = (5,5)
    loss_fig_snr = plt.figure(figsize = fig_dim)
    plt.errorbar(range(0, len(loss_mean)), np.mean(loss_mean_array, axis = 0), yerr = stats.sem(loss_mean_array, axis = 0))
    
    plt.figure()
    plt.errorbar(range(0, len(loss_mean)), np.mean(similarity_ws_r, axis = 0), yerr = stats.sem(similarity_ws_r, axis = 0))
    plt.title('similarity wake-sleep vs reinforce')
    
    
    cosine_similarity_ws_bp_fig_snr, ax = plt.subplots(figsize = fig_dim)
    plt.errorbar(range(0, len(loss_mean)), np.mean(cosine_similarity_ws_r, axis = 0), yerr = stats.sem(cosine_similarity_ws_r, axis = 0))
    plt.ylim([0,1])
    plt.title('cosine similarity wake_sleep vs. backprop')
    
    norm_ws_bp_snr, ax_2 = plt.subplots(figsize = fig_dim)
    ax_2.errorbar(range(0, len(loss_mean)), np.mean(gradient_norm_ws, axis = 0), yerr = stats.sem(gradient_norm_ws, axis = 0), color = 'k')
    ax_2.errorbar(range(0, len(loss_mean)), np.mean(gradient_norm_reinforce, axis = 0), yerr = stats.sem(gradient_norm_ws, axis = 0), color = 'r')
    ax_2.set_ylabel('norm')
    
    
    snr_fig = plt.figure(figsize = fig_dim)
    plt.errorbar(range(0, len(loss_mean)), np.mean(mean_snr_ws, axis = 0), yerr = stats.sem(mean_snr_ws, axis = 0))
    plt.errorbar(range(0, len(loss_mean)), np.mean(mean_snr_r, axis = 0), yerr = stats.sem(mean_snr_r, axis = 0))
    plt.yscale('log')
    plt.legend(('ws','r','bp'))
    plt.title('snr through learning')
    if image_save:
        loss_fig_snr.savefig(plot_save_folder+'loss_snr.pdf', format = 'pdf')
        norm_ws_bp_snr.savefig(plot_save_folder+'norm_ws_bp_snr.pdf', format = 'pdf')
        cosine_similarity_ws_bp_fig_snr.savefig(plot_save_folder + 'cosine_similarity_fig_snr.pdf', format = 'pdf')
        snr_fig.savefig(plot_save_folder+'snr_fig.pdf', format = 'pdf')


#%% Plots for varying the OU process time constant
if exp_params.mode == 'time_constant':
    fig_dim = (5,5)
    loss_mean_aggregate = []
    corr = np.zeros((10,))
    for ii in range(1,11):
        filename = 'impression_tc_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_aggregate.append(loss_mean)
        
        plt.figure(figsize = fig_dim)
        plt.plot(wake_sleep_sequence.latent[1, 0:400])
        plt.plot(wake_sequence.latent[1,0:400])
        
        corr[ii-1] = np.corrcoef(wake_sleep_sequence.latent.flatten(), wake_sequence.latent.flatten())[0,1]
        
    
        plt.figure(figsize = fig_dim)
        plt.scatter(wake_sequence.latent[1,0:1000], wake_sleep_sequence.latent[1,0:1000])
        
    loss_mean_total = np.vstack(loss_mean_aggregate)
    loss_fig_tc = plt.figure(figsize = fig_dim)
    plt.plot(loss_mean_total.T)
    plt.legend(exp_params.dt_list)
    
    plt.figure()
    plt.scatter(exp_params.dt_list, corr)
    plt.xlabel('dt')
    plt.ylabel('corr')
    
#%% Plots for varying the switch period

if exp_params.mode == 'switch_period':
    fig_dim = (5,5)
    loss_mean_aggregate = []
    corr = np.zeros((11,))
    for ii in range(1,12):
        filename = 'impression_sp_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_aggregate.append(loss_mean)
        
        if ii == 1 or ii == 11:
            sp_accuracy_fig = plt.figure(figsize = fig_dim)
            if ii == 1:
                idx = 0
                y_limit = 1.7
            else:
                idx = 0
                y_limit = 1.7
            plt.plot(wake_sleep_sequence.latent[idx, 300:500])
            plt.plot(wake_sequence.latent[idx,300:500])
            plt.ylim([-y_limit,y_limit])
            
            plt.figure(figsize = fig_dim)
            plt.scatter(wake_sequence.latent[1,0:1000], wake_sleep_sequence.latent[1,0:1000])
            
            if image_save:
                sp_accuracy_fig.savefig(plot_save_folder+'sp_accuracy_'+str(ii)+'.pdf', format = 'pdf')
        corr[ii-1] = np.corrcoef(wake_sleep_sequence.latent.flatten(), wake_sequence.latent.flatten())[0,1]

    loss_mean_total = np.vstack(loss_mean_aggregate)
    loss_fig_tc = plt.figure(figsize = fig_dim)
    plt.plot((loss_mean_total[[0,10],:].T))
    plt.legend(exp_params.switch_period_list[[0,10]])
    
    sp_corr_fig = plt.figure(figsize = fig_dim)
    plt.scatter(exp_params.switch_period_list+1, corr)
    plt.xlabel('switch period')
    plt.ylabel('corr')
    if image_save:
        loss_fig_tc.savefig(plot_save_folder+'sp_loss.pdf', format = 'pdf')
        sp_corr_fig.savefig(plot_save_folder+'sp_corr.pdf', format = 'pdf')

#%% Plots for varying the dimensionality
    
if exp_params.mode == 'dimensionality':
    fig_dim = (5,5)
    loss_mean_aggregate = []
    loss_mean_bp = []
    loss_mean_imp = []
    loss_mean_r = []
    for ii in range(1,11):
        filename = 'impression_d_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_aggregate.append(loss_mean)
        
    for ii in range(1,6):
        filename = 'impression_d_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_imp.append(np.mean(wake_sequence.loss))
        
    for ii in range(6,11):
        filename = 'impression_d_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_r.append(np.mean(wake_sequence.loss))
        
    for ii in range(1,6):
        filename = 'impression_d_bp_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_bp.append(np.mean(loss_test))
        
    loss_mean_total = np.vstack(loss_mean_aggregate)
    loss_fig_tc = plt.figure(figsize = fig_dim)
    plt.plot(loss_mean_total[0:5,:].T)
    
    loss_fig_tc = plt.figure(figsize = fig_dim)
    plt.plot(loss_mean_total[5:10,:].T)
    
    dimensionality_fig = plt.figure(figsize = fig_dim)
    plt.scatter(2* 2**np.arange(1,6), loss_mean_imp)
    plt.scatter(2 * 2**np.arange(1,6), loss_mean_r)
    plt.scatter(2 * 2**np.arange(1,6), loss_mean_bp)
    plt.legend(('impression', 'reinforce', 'bp'))#, 'initial loss'))
    plt.xlabel('latent/neural dimension')
    plt.ylabel('asymptotic loss')
    if image_save:
        dimensionality_fig.savefig(plot_save_folder+'dimensionality.pdf', format = 'pdf')


#%% Plots for learning rate optimization
    
if exp_params.mode == 'lr_optim':
    fig_dim = (5,5)
    loss_mean_aggregate = []
    for ii in range(1,41):
        filename = 'impression_lr_' + str(ii)
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_aggregate.append(loss_mean)
    
    loss_mean_total = np.vstack(loss_mean_aggregate)
    loss_fig_tc = plt.figure(figsize = fig_dim)
    plt.plot(loss_mean_total[0:20,:].T)
    
    loss_fig_tc = plt.figure(figsize = fig_dim)
    plt.plot(loss_mean_total[20:40,:].T)

#%% plots for the spoken digits dataset
def to_DB(X):
    return (X)*22.107208 - 80

if exp_params.mode == 'Vocal_Digits':
    if not(exp_params.local_plot):
        filename = 'vocal_digits_10'
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
        loss_mean_reinforce = loss_mean
        
        filename = 'vocal_digits_1'
        path = os.getcwd() + load_folder + filename
        data = load_file(path)
        unpack_loaded_data(data)
    else:
        loss_mean_2 = 0
    
    fig_dim = (5,5)
    vocal_loss_fig = plt.figure(figsize = fig_dim)
    plt.ylim([100000, 3500000])
    plt.plot(loss_mean[0::])
    plt.ylabel('loss')
    plt.xlabel('epoch_num')
    
    plt.yscale('log')
    if not(exp_params.local_plot):
        plt.plot(loss_mean_reinforce[0::])
    plt.legend(('impression', 'reinforce'))
    
    frequencies = librosa.core.mel_frequencies()[0:85] #extract the frequencies used for the mel scale
    frequencies_rd = np.around(frequencies, decimals = -2).astype(int)
    tickrange = range(0,85,20)
    
    
    vocal_reproduction, ax = plt.subplots(1,2, figsize = fig_dim)
    sample = to_DB(data_test[:,0:18])
    sample_reproduction = to_DB(network.l0.W_out @ wake_sequence.latent[0:network.l1.N,0:18])
    img_1 = ax[0].imshow(sample[0:85], vmin = np.min(sample_reproduction), vmax = np.max(sample_reproduction), origin = 'lower')
    img_2 = ax[1].imshow(sample_reproduction[0:85,:], vmin = np.min(sample_reproduction), vmax = np.max(sample_reproduction), origin = 'lower')
    
    ax[0].set_yticks(range(0,85,20))
    ax[0].set_yticklabels(frequencies_rd[tickrange])
    ax[1].set_yticks(range(0,85,20))
    ax[1].set_yticklabels(frequencies_rd[tickrange])
    vocal_reproduction.colorbar(img_1, format='%+2.0f dB')
    
    fig_dim_2 = (10,5)
    generative_comparison_fig, ax = plt.subplots(1,2, figsize = fig_dim_2)
    data_dist = to_DB(data_test)
    gen_dist = to_DB(network.l0.W_out @ gen_sim.latent[0:network.l1.N,:])
    data_cov = np.corrcoef(data_dist)
    gen_cov = np.corrcoef(gen_dist)
    upper_lim = np.mean(data_cov[0:85,0:85]) + 3*np.std(data_cov[0:85,0:85])
    lower_lim = np.mean(data_cov[0:85,0:85]) - 3*np.std(data_cov[0:85,0:85])
    img_1 = ax[0].imshow(data_cov[0:85,0:85], vmin = 0, vmax = 1, cmap = 'viridis', origin = 'lower')
    img_2 = ax[1].imshow(gen_cov[0:85, 0:85], vmin = 0, vmax = 1, cmap = 'viridis', origin = 'lower')
    generative_comparison_fig.colorbar(img_2, ax = ax[1])
    generative_comparison_fig.colorbar(img_1, ax = ax[0])
    ax[0].set_yticks(range(0,85,20))
    ax[0].set_yticklabels(frequencies_rd[tickrange])
    ax[0].set_xticks(range(0,85,20))
    ax[0].set_xticklabels(frequencies_rd[tickrange])
    
    ax[1].set_yticks(range(0,85,20))
    ax[1].set_yticklabels(frequencies_rd[tickrange])
    ax[1].set_xticks(range(0,85,20))
    ax[1].set_xticklabels(frequencies_rd[tickrange])
    print(cosine_similarity(data_cov, gen_cov))
    
    #%%
    vocal_gen_fig, ax = plt.subplots(figsize = fig_dim)
    gen_sample = network.l0.W_out @ gen_sim.latent[0:network.l1.N,4340:4370]
    img = plt.imshow(to_DB(gen_sample[0:85,:]), origin = 'lower')
    frequencies = librosa.core.mel_frequencies()[0:85] #extract the frequencies used for the mel scale
    frequencies_rd = np.around(frequencies, decimals = -2).astype(int)
    tickrange = range(0,85,20)
    ax.set_yticks(range(0,85,20))
    ax.set_yticklabels(frequencies_rd[tickrange])
    vocal_gen_fig.colorbar(img, format='%+2.0f dB')
    #%%
    vocal_acf_fig, ax = plt.subplots(figsize = fig_dim)
    sm.graphics.tsa.plot_acf(wake_sequence.latent[5,:], ax = ax, lags=40, use_vlines = False)
    sm.graphics.tsa.plot_acf(gen_sim.latent[5,:], ax = ax, lags = 40, use_vlines = False)
    plt.legend(('wake', 'sleep'))
    plt.xlabel('time lag')
    plt.ylabel('autocorrelation')
    
    
    #%% Saving
    if image_save:
        vocal_gen_fig.savefig(plot_save_folder+'vocal_gen_fig.pdf', format = 'pdf')
        vocal_reproduction.savefig(plot_save_folder+'vocal_reproduction.pdf', format = 'pdf')
        vocal_loss_fig.savefig(plot_save_folder+'vocal_loss_fig.pdf', format = 'pdf')
        vocal_acf_fig.savefig(plot_save_folder+'vocal_acf_fig.pdf', format = 'pdf')
        generative_comparison_fig.savefig(plot_save_folder+'vocal_gen_comparison_fig.pdf', format = 'pdf')