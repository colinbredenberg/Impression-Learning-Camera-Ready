# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:23:25 2021

@author: colin
"""

import librosa
import librosa.display

import re
import pickle
import random

import os
import numpy as np

#Change path to the location of your Free Spoken Digits Dataset folder
#Change output to the location of your Impression Learning code
folder_path = "your_path\\free-spoken-digit-dataset-master\\recordings\\"
output_path = "your_impression_learning_path\\"
spectrogram_list_train = []
spectrogram_labels_train = []
spectrogram_list_test = []
spectrogram_labels_test = []

#%% Generate spectrograms
ctr_max = len(os.listdir(folder_path))
ctr = 0
for filename in os.listdir(folder_path):
    ctr = ctr + 1
    print(ctr/ctr_max * 100)
    file, sample_rate = librosa.load(folder_path + filename)
    
    spectrogram = librosa.feature.melspectrogram(file, sr = sample_rate)
    spectrogram = librosa.power_to_db(spectrogram, ref = np.max)
    
    if re.findall("_[0-4].wav", filename): #add the 1st five instances of a speaker to the test set, the others to the train set
        spectrogram_list_test.append(spectrogram)
        label = int(re.findall("\d+_", filename)[0][0:-1]) * np.ones((1,spectrogram.shape[1]))
        spectrogram_labels_test.append(label)
    else:
        spectrogram_list_train.append(spectrogram)
        label = int(re.findall("\d+_", filename)[0][0:-1]) * np.ones((1,spectrogram.shape[1]))
        spectrogram_labels_train.append(label)

#%% save the files


#shuffle the train and test lists in exactly the same way
idx = list(range(0, len(spectrogram_list_test)))
random.shuffle(idx)
spectrogram_list_test = list(spectrogram_list_test[i] for i in idx)
spectrogram_labels_test = list(spectrogram_labels_test[i] for i in idx)

idx = list(range(0, len(spectrogram_list_train)))
random.shuffle(idx)
spectrogram_list_train = list(spectrogram_list_train[i] for i in idx)
spectrogram_labels_train = list(spectrogram_labels_train[i] for i in idx)

train_set = np.concatenate(spectrogram_list_train, axis = 1)
train_labels = np.concatenate(spectrogram_labels_train, axis = 1)
test_set = np.concatenate(spectrogram_list_test, axis = 1)
test_labels = np.concatenate(spectrogram_labels_test, axis = 1)

data = {'train_set': train_set,
        'train_labels': train_labels,
        'test_set': test_set,
        'test_labels': test_labels}

with open('spoken_digits_dataset', 'wb') as f:
    pickle.dump(data, f)


