'''
This file contains the basic functions of this repo for further ML tasks
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm.notebook import tqdm

# The dataset directory
file_dir = './aqi_dataset.csv'

def df_prep(file):
    df = pd.read_csv(file)
    # Delete unnecessary columns
    df = df.drop(columns=['Country', 'City', 'AQI Category', 'CO AQI Category', 'Ozone AQI Category', 'NO2 AQI Category', 'PM2.5 AQI Category'])
    # Randomly shuffle the rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split training set and test set with 8-2 ratio
    train = df.sample(frac=0.8,random_state=42)
    test = df.drop(train.index)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test

def df_prep_full(file):
    df = pd.read_csv(file)
    return df

def make_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
        
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error')
    plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
    plt.legend()
