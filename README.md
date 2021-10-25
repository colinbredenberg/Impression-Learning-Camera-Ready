# Impression-Learning-Camera-Ready
Camera ready code repo for the NeuRIPS 2021 paper: "Impression learning: Online representation learning with synaptic plasticity," by Colin Bredenberg, Benjamin S. H. Lyo, Eero P. Simoncelli, and Cristina Savin.


# Requirements

-numpy

-time

-os

-copy

-deepcopy

-re

-matplotlib.pyplot

-pickle

-scipy

For the Free Spoken Digits Dataset simulations:
-librosa

For the backpropagation implementation:
-pytorch (https://pytorch.org/)

# Instructions
In what follows, we will summarize how to reproduce the results of our paper with the code.
Though some of our results require a cluster, our primary results (training + figure generation) can be completed
in ~5-10 minutes on a personal computer.

**Experimental Parameters (il_exp_params.py)**
This file specifies the particular type of simulation to run, and selects simulation hyperparameters accordingly.

To generate Figure 1 (~5 min runtime): set mode = 'standard'. This can be run on a local computer.

To generate Figure 2: set mode = 'SNR' (Fig. 2a-c) or set mode = 'dimensionality' (Fig. 2d). This will require a cluster.

To generate Figure 3: set mode = 'switch_period'. This will require a cluster.

To generate Figure 4 (~8 min runtime): set mode = 'Vocal_Digits'. This can be run on a local computer. Running this simulation will require librosa, as well as our preprocessed dataset (See Preprocessing FSDD).

To save data after a simulation, set save = True

**Running a simulation (impression_learning.py)**
To run a simulation, simply run impression_learning.py after setting experimental parameters appropriately.

**Plotting (il_plot_generator.py)**
To plot data after a simulation, simply run il_plot_generator.py. We ran these files consecutively in an IDE (e.g. Spyder).
To save the results of a simulation, set image_save = True, which will save images in your local directory.


**Backpropagation controls:**
We used Pytorch to separately train our backpropagation control, which has its own experimental parameters.

Experimental Parameters (il_exp_params_bp.py): array_num determines the dimensionality of the latent space.

Running a simulation and generating plots (il_backprop.py):

To run a simulation, simply run il_backprop.py. Plots for the chosen dimensionality will automatically be produced at the end of simulation.

**Preprocessing the Free Spoken Digits Dataset (FSDD) (il_fsdd_preprocessing.py)**
For Figure 4 we generate spectrograms from the FSDD. Generating this plot will require our preprocessed data, run on the data from the FSDD (https://github.com/Jakobovski/free-spoken-digit-dataset). To preprocess the data, set your folder path to the location of your downloaded FSDD recordings folder, and set your output path to the location of your downloaded Impression Learning code. All that remains is to run the il_fsdd_preprocessing.py file (~5 min runtime).
