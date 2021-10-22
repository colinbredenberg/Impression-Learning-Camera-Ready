import torch
from torch import nn
from torch import linalg as la
from torch import optim
import numpy as np
import il_exp_params_bp as exp_params
import os
import pickle
if exp_params.local:
    import matplotlib.pyplot as plt
class HelmHoltzCell(nn.Module):
    def __init__(self, n_latent, n_in, nonlinearity):
        """ Requirements
            n_latent: number of neurons
            n_in: number of input dimensions
            W_in: n_latent x n_in feedforward weight matrix
            sigma_obs: input noise for the recognition model
            sigma_latent: latent noise for the recognition model
            sigma_obs_gen: noise added to inputs for the generative model
            sigma_latent_gen: noise added to latent variables for the generative model
            nonlinearity: function for the nonlinearity
        """
        super().__init__()

        self.n_latent = n_latent
        self.n_in = n_in

        # weight matrices to train
        self.W_in = nn.Parameter(torch.rand(self.n_latent, self.n_in), requires_grad=True)
        self.W_out = nn.Parameter(torch.rand(self.n_in, self.n_latent), requires_grad=True)
        self.D_r = nn.Parameter(torch.rand(self.n_latent,), requires_grad=True)

        # parameters to initialise
        self.sigma_obs_inf = nn.Parameter(torch.rand(1), requires_grad=False)
        self.sigma_latent_inf = nn.Parameter(torch.rand(1), requires_grad=False)
        self.sigma_latent_gen = nn.Parameter(torch.rand(1), requires_grad=False)
        self.sigma_obs_gen = nn.Parameter(torch.rand(1), requires_grad=False)

        self.latent_inf = torch.rand(self.n_latent,)
        self.latent_gen = torch.rand(self.n_latent,)
        self.latent_mean_inf = torch.rand(self.n_latent,)
        self.latent_mean_gen = torch.rand(self.n_latent,)
        self.latent = torch.rand(self.n_latent,)

        self.s_mean_gen = torch.rand(self.n_in,)
        self.s_mean_inf = torch.rand(self.n_in,)
        self.s = torch.rand(self.n_in,)

        self.nl = nonlinearity

        # phase
        self.Lambda = nn.Parameter(torch.ones(1), requires_grad=False)
        self.phase = "wake"
        self.set_phase(self.phase)

        self.k = 1

        self.weights_initialized = False

    def set_phase(self, phase):
        """sets the phase (wake or sleep) for the network"""
        self.phase = phase
        if phase == "wake":
            self.Lambda = nn.Parameter(torch.ones(1))
        elif phase == "sleep":
            self.Lambda = nn.Parameter(torch.zeros(1))
        return

    def toggle_phase(self):
        """toggles the phase that the network is in"""
        if self.phase == "wake":
            self.set_phase("sleep")
        elif self.phase == "sleep":
            self.set_phase("wake")
        return

    def set_k(self, prev_lambda):
        delta = int(prev_lambda == self.Lambda.item())
        self.k = (1 - delta) * self.Lambda

    def forward(self, data, prev_lambda, prev_latent):
        assert self.weights_initialized, 'Initialize weights before training'

        self.set_k(prev_lambda)

        # generate new latent_gen
        transition_mat = (1 - self.k) * torch.diag_embed(self.D_r) + self.k * torch.eye(self.n_latent)
        self.latent_mean_gen = torch.matmul(transition_mat, prev_latent)
        latent_noise_gen = torch.normal(mean=0., std=self.sigma_latent_gen.item(), size=(self.n_latent,))
        self.latent_gen = self.latent_mean_gen + latent_noise_gen

        # sleep phase updates
        #self.latent_gen = torch.normal(mean=0., std=self.sigma_latent_gen.item(), size=(self.n_latent,))
        h_gen = torch.matmul(self.W_out, self.latent_gen)
        self.s_mean_gen = self.nl.f(h_gen)
        obs_noise_gen = torch.normal(mean=0., std=self.sigma_obs_gen.item(), size=(self.n_in,))
        s_gen = self.s_mean_gen + obs_noise_gen

        # wake phase updates
        obs_noise_inf = torch.normal(mean=0., std=self.sigma_obs_inf.item(), size=(self.n_in,))
        self.s_mean_inf = data
        s_inf = self.s_mean_inf + obs_noise_inf
        h_inf = torch.matmul(self.W_in, s_inf)
        self.latent_mean_inf = self.nl.f(h_inf)
        latent_noise_inf = torch.normal(mean=0., std=self.sigma_latent_inf.item(), size=(self.n_latent,))
        self.latent_inf = self.latent_mean_inf + latent_noise_inf

        # delta determines whether the network is in the sleep phase or the wake phase
        self.s = self.Lambda * s_inf + (1 - self.Lambda) * s_gen
        self.latent = self.Lambda * self.latent_inf + (1 - self.Lambda) * self.latent_gen


class HelmholtzModel(nn.Module):
    def __init__(
            self,
            n_neurons, n_in, n_latent,
            sigma_obs_inf, sigma_latent_inf, sigma_obs_gen, sigma_latent_gen,
            learning_rate, switch_period, nonlinearity
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_latent = n_latent

        self.W_in = torch.from_numpy(np.random.normal(loc=0., scale=1/self.n_in, size=(self.n_neurons, self.n_in))).float()
        self.W_out = torch.from_numpy(np.random.normal(loc=0., scale=1/self.n_neurons, size=(self.n_in, self.n_neurons))).float()
        self.D_r = torch.ones(self.n_latent) * 0.6
        
        self.sigma_obs_inf = sigma_obs_inf
        self.sigma_latent_inf = sigma_latent_inf
        self.sigma_obs_gen = sigma_obs_gen
        self.sigma_latent_gen = sigma_latent_gen

        self.nl = nonlinearity

        self.learning_rate = learning_rate
        self.switch_period = switch_period
        self.switch_counter = 0  # counter that keeps track of how many data points since last switch

        self.cell = HelmHoltzCell(n_in=self.n_in, n_latent=self.n_latent, nonlinearity=self.nl)

        self.loss = 0.

    def calculate_loss(self):
        # during wake
        Lp = la.norm(self.cell.latent - self.cell.latent_mean_gen)**2 / self.cell.sigma_latent_gen**2 + \
             la.norm(self.cell.s -
                     self.nl.f(torch.matmul(self.cell.W_out, self.cell.latent_inf)))**2 / self.cell.sigma_obs_gen**2 -\
             la.norm(self.cell.latent - self.cell.latent_mean_inf)**2 / self.cell.sigma_latent_inf**2 - \
             la.norm(self.cell.s - self.cell.s_mean_inf)**2/self.cell.sigma_latent_inf**2
        # during sleep
        Lq = la.norm(self.cell.latent -
                     self.nl.f(torch.matmul(self.cell.W_in, self.cell.s)))**2 / self.cell.sigma_latent_inf**2 #-\
             #la.norm(self.cell.latent - self.cell.latent_mean_gen)**2/self.cell.sigma_latent_gen**2 + \
             #la.norm(self.cell.s - self.cell.s_mean_inf)**2/self.cell.sigma_obs_inf**2 - \
             #la.norm(self.cell.s - self.cell.s_mean_gen)**2/self.cell.sigma_obs_gen**2

        if self.cell.phase == "wake":
            self.cell.W_in.requires_grad = True
            self.cell.W_out.requires_grad = True
            self.cell.D_r.requires_grad = True
        elif self.cell.phase == "sleep":
            self.cell.W_in.requires_grad = True
            self.cell.W_out.requires_grad = True
            self.cell.D_r.requires_grad = True

        self.loss = torch.matmul(self.cell.Lambda, Lp) + torch.matmul((1 - self.cell.Lambda), Lq)

    def forward(self, data, train=True, phase_switch = True):
        T = len(data[1])
        latents = np.zeros((self.n_latent, T))  # record the latent neuron activations
        losses = np.zeros((1, T))  # record the loss through time

        # variables for printing average loss
        avg_loss = 0.
        avg_period = 2000.
        bptt_period = 10
        optimizer = optim.SGD(
            [self.cell.W_in, self.cell.W_out], lr=self.learning_rate, momentum=0
        )
        optimizer.add_param_group(
            {
                'params': self.cell.D_r,
                'lr': self.learning_rate * self.sigma_latent_gen**2 / self.sigma_latent_inf**2,
                'momentum': 0,
            }
        )

        prev_lambda = 0
        prev_latent = self.cell.latent.detach()

        for tt in range(0, T):  # loop through all of the data
            self.cell(data[:, tt], prev_lambda, prev_latent)  # get activations for one data point

            prev_lambda = self.cell.Lambda.detach().item()
            prev_latent = self.cell.latent.detach()

            # update the learning variables/parameters
            if train:
                optimizer.zero_grad()  # reset the gradients
                self.calculate_loss()
                self.loss.backward()
# =============================================================================
#                 if (np.mod(tt, bptt_period) == 0):
#                     self.loss.backward()  # propagate gradients backwards
#                     prev_latent = self.cell.latent.detach()
#                 else:
#                     self.loss.backward(retain_graph = True)
# =============================================================================
                optimizer.step()

            elif not train:
                #self.cell.set_phase('wake')  # keep the network in the wake phase unless being trained
                self.calculate_loss()
            
            if phase_switch:
                if self.switch_period > 0:
                    self.switch_counter += 1  # increment the switch counter
                    if self.switch_counter >= self.switch_period:  # if the switch counter is greater than the period, toggle the phase
                        self.cell.toggle_phase()  # switch from wake to sleep or vice versa
                        self.switch_counter = 0  # reset the counter
            else:
                self.cell.set_phase('wake')

            # keep record of neural activations and loss
            latents[:, tt] = self.cell.latent.detach()
            losses[:, tt] = self.loss.detach().item()

            if tt % avg_period == 0 and tt > 0:
                print('t:', tt)
                avg_loss = np.sum(losses[:, tt - int(avg_period):tt]) / avg_period
                print('avg loss:', avg_loss)
                avg_loss = 0

        print('D_r', self.cell.D_r.detach())
        transition_mat = torch.diag(self.cell.D_r.detach())
# =============================================================================
#         plt.title('Transition matrix')
#         plt.imshow(transition_mat)
#         plt.colorbar()
#         plt.show()
# =============================================================================

        return latents, losses

    def init_weights(self):
        state_dict = {
            'W_in': self.W_in,
            'W_out': self.W_out,
            'D_r': self.D_r,
            'sigma_obs_inf': torch.ones(1) * self.sigma_obs_inf,
            'sigma_obs_gen': torch.ones(1) * self.sigma_obs_gen,
            'sigma_latent_inf': torch.ones(1) * self.sigma_latent_inf,
            'sigma_latent_gen': torch.ones(1) * self.sigma_latent_gen,
            'Lambda': torch.ones(1) * 1,
        }

        self.cell.load_state_dict(state_dict)
        self.cell.weights_initialized = True
        print('weights initialized!')


# Generate simulated inputs (Static FA)
def simulate_data(n_latent, n_out, n_sample, mixing_matrix, sigma_latent=1, sigma_out=0.01):
    """simulate_data: generates data points for the Helmholtz Machine to learn on
    n_latent: number of latent states
    n_out: number of observed dimensions
    n_sample: number of samples to draw
    mixing_matrix: n_out x n_latent matrix mapping latent variables to observed
    sigma_latent: latent noise (default 1) in p(z)
    sigma_out: observation noise in p(s|z)"""

    # draw samples from the latent var
    latent = torch.normal(mean=0., std=sigma_latent, size=(n_latent, n_sample))
    # generate observation noise
    obs_noise = torch.normal(mean=0., std=sigma_out, size=(n_out, n_sample))

    # produce observations from the latent variables and the noise
    data = torch.matmul(mixing_matrix, latent) + obs_noise
    return data


def simulate_temporal_data(n_latent, n_out, n_sample, mixing_matrix, transition_matrix, sigma_latent=1, sigma_out=0.01):
    """simulate_data: generates data points for the Helmholtz Machine to learn on
    n_latent: number of latent states
    n_out: number of observed dimensions
    n_sample: number of samples to draw
    mixing_matrix: n_out x n_latent matrix mapping latent variables to observed
    sigma_latent: latent noise (default 1)
    sigma_out: observation noise"""
    # draw samples from the latent var
    print('simulating temporal data...')
    latent_noise = torch.from_numpy(np.random.normal(scale=sigma_latent, size=(n_latent, n_sample))).float()
    latent = torch.zeros((n_latent, n_sample))
    for ii in range(0, n_sample):
        if ii > 0:
            latent[:, ii] = torch.matmul(transition_matrix, latent[:, ii-1]) + latent_noise[:, ii]
        else:
            latent[:, ii] = latent_noise[:, ii]
    # generate observation noise
    obs_noise = torch.from_numpy(np.random.normal(scale=sigma_out, size=(n_out, n_sample))).float()
    # produce observations from the latent variables and the noise
    data = torch.matmul(mixing_matrix, latent)  # + obs_noise
    print('temporal data simulated!')
    return data, latent


class Function:
    """Defines a function and its derivative.
    Attributes:
        f (function): An element-wise differentiable function that acts on a
            1-d numpy array of arbitrary dimension. May include a second
            argument for a label, e.g. for softmax-cross-entropy.
        f_prime (function): The element-wise derivative of f with respect to
            the first argument, must also act on 1-d numpy arrays of arbitrary
            dimension.
    """
    def __init__(self, f, f_prime):
        """Inits an instance of Function by specifying f and f_prime."""
        self.f = f
        self.f_prime = f_prime


def tanh_(z):
    return torch.tanh(z)


def tanh_derivative(z):
    return 1 - torch.tanh(z) ** 2


def reformat_data(loss, n_sample, switch_period, train):
    if train:
        loss_split = loss.reshape((int(n_sample / (switch_period * 2)), switch_period * 2))
        loss_wake = np.mean(loss_split[:, 0:switch_period], axis=1)
        loss_sleep = np.mean(loss_split[:, switch_period::], axis=1)
        mean_len = 100
        loss_wake = np.mean(loss_wake.reshape([int(len(loss_wake) / mean_len), mean_len]), axis=1)
        loss_sleep = np.mean(loss_sleep.reshape([int(len(loss_sleep) / mean_len), mean_len]), axis=1)
        loss = (loss_wake, loss_sleep)
    else:
        loss = np.mean(loss.reshape([int(len(loss[0]) / n_sample), n_sample]), axis=1)
    return loss


if __name__ == '__main__':
    '''Run simulation and perform comparisons'''
    np.random.seed(120994)
    torch.manual_seed(120994)
    
    array_num = exp_params.array_num
    # simulate the data
    n_latent = exp_params.n_latent
    n_out = exp_params.n_out
    n_in = exp_params.n_in
    n_neurons = exp_params.n_neurons # number of latent dimensions for neurons
    n_sample = exp_params.n_sample  # 2000000  # number of data points for the train dataset
    n_test = exp_params.n_test  # 30000  # number of data points for the test dataset
    dt = 0.1  # time step for the data OU process
    sigma_latent_data = 0.5 * np.sqrt(dt)
    mixing_matrix = torch.from_numpy(np.random.normal(scale=1/n_latent, size=(n_in, n_latent))).float()  # observation matrix
    transition_matrix = (1 - sigma_latent_data**2) * torch.eye(n_latent)

    # static datasets
    # data_train = simulate_data(n_latent, n_out, n_sample, mixing_matrix, sigma_latent=1, sigma_out=0.01)
    # data_test = simulate_data(n_latent, n_out, n_test, mixing_matrix, sigma_latent=1, sigma_out=0.01)

    # temporal datasets
    data_train, data_latent_train = simulate_temporal_data(
        n_latent, n_out, n_sample,
        mixing_matrix, transition_matrix,
        sigma_latent=sigma_latent_data, sigma_out=0.01
    )
    data_test, data_latent_test = simulate_temporal_data(
        n_latent, n_out, n_test,
        mixing_matrix, transition_matrix,
        sigma_latent=sigma_latent_data, sigma_out=0.01
    )

    # build neural network
    tanh = Function(tanh_, tanh_derivative)
    sigma_latent_inf = exp_params.sigma_latent  # latent noise for the network (sigma_latent)
    sigma_obs_gen = exp_params.sigma_obs_gen  # latent noise for the network
    sigma_obs_inf = exp_params.sigma_in
    
    sigma_latent_gen = sigma_latent_data
    
    learning_rate = exp_params.learning_rate* 10 * 0.01**2 
    switch_period = exp_params.switch_period  # number of samples taken before switching from wake to sleep

    network = HelmholtzModel(
        n_neurons, n_in, n_latent, sigma_obs_inf,
        sigma_latent_inf, sigma_obs_gen, sigma_latent_gen,
        learning_rate, switch_period, nonlinearity=tanh
    )
    network.init_weights()

    # run the training simulation
    latent_train, loss = network(data_train, train=True)
    
    # run the test simulation
    latent_test, loss_test = network(data_test, train=False, phase_switch = False)
    
    # run an alternating test simulation
    latent_test_switch, loss_test_switch = network(data_test, train=False, phase_switch = True)
    #%%
    # reformat data for plotting
    #(loss_wake, loss_sleep) = reformat_data(loss, n_sample, switch_period, train=True)
    loss_reformat = reformat_data(loss, 1000, switch_period, train = False)
    if exp_params.local and exp_params.local_plot:
        plt.figure()
        #plt.plot(loss_sleep)
        #plt.plot(loss_wake)
        #plt.legend(('sleep phase', 'wake phase'))
        plt.plot(loss_reformat)
        plt.title('loss through training')
        plt.show()
    
        
    
        # Plot the input and targets
        W_out = network.cell.W_out.detach().numpy()
        prediction = np.ndarray.flatten(W_out @ latent_test)
        true = np.ndarray.flatten(data_test.numpy())
    
        plt.figure()
        plt.scatter(true, prediction)
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.plot([-1, 1], [-1, 1], 'k')
        plt.xlabel('ground truth')
        plt.ylabel('prediction')
        plt.show()
    
        # See if the network has successfully whitened its latent variables
        cov = np.cov(latent_test)
        plt.imshow(cov)
        plt.colorbar()
        plt.show()
    
    #%% Save
    if exp_params.save:
    #lump all of the results into one dictionary
        if exp_params.mode in ('dimensionality'):
            result = {'network': network,# 'sim': sim, 
                      'loss_test': loss_test,
                      'data_test': data_test,
                      'latent_test': latent_test}
            if exp_params.mode == 'dimensionality':
                filename = '/impression_d_bp_'

        #save the whole dictionary
        if exp_params.local:
            save_path = os.getcwd() + 'anonymous_filepath_3' + 'impression_data' + str(array_num)
        else:
            save_path = os.getcwd() + filename + str(array_num)
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)