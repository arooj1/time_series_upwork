# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:17:19 2021

@author: quresa9
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
from tensorflow.keras.models import model_from_json
import test_class_functions as tsf 


'''
Purpose of this file is to test output file results from `test_class_functions.py` using training model from `main.py`

'''

# STEP 1: Define Paths 
__file__ = 'time_series_upwork'
ROOT = os.path.abspath(__file__)
model_name = 'model.json'
model_weights = 'model.h5'

# STEP 2: Training model values required for preparing test dataset. 
# These values are received form `main.py`

training_mean = 42.438353335806646
training_std  = 28.077122281262515
threshold = 0.136253187635769

# STEP 3: Initialize `test_class_function`

tclass = tsf.test_data(model_name = model_name,
             model_weights = model_weights,
             training_mean = training_mean,
             training_std = training_std,
             training_threshold = threshold)
print('Class - `test_class_function` initialized')

# STEP 4: Load Test Data 
master_url_root = "D:/AAQ/Upwork/anomaly_detection/time_series_upwork/"
df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = os.path.join(master_url_root, df_daily_jumpsup_url_suffix)
test_data = pd.read_csv(df_daily_jumpsup_url, parse_dates=True, index_col="timestamp")
print('TEST DATA LOADED')
print(test_data.head(5))
a = test_data.copy()

# STEP 5: Predicting Test data for Anomaly Detection 
results = tclass(test_data)

