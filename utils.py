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
    df = df.drop(columns=['Country', 'AQI Category', 'CO AQI Category', 'Ozone AQI Category', 'NO2 AQI Category', 'PM2.5 AQI Category'])
    
    # WIP: normalization of the data
    
    return df
    