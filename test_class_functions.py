# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:32:00 2021

@author: quresa9
"""

"""
### Prepare test data
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
import os

class test_data:
    '''
    PURPOSE: It will predict anomaly based on the following variables of the trained model.
    
    1-NORMAIZE_TEST_DATA: normalise test dataset using mean and standard deviation value from the training
                            dataset. 
                            
    2-TRAINED TRESHOLD (for anomaly detection): this is the maximum value of MAE loss of the training 
                                                dataset.
                                                
    3-MAE Loss (TEST DATA): It will calculate MAE loss of the test_dataset.
    
    4-ADD INFORMATION: Add following information in front of each data sampel of test dataset group. 
    
        - **group_id** id allocated to one set of dataset

        - **threshold** is max of train_mae_loss

        - **MAE value** of individual time-sample of test dataset

        - **Anomaly** binary output of each time-sample of test dataset

    RETURN : Dataframe
    
    '''
    
    def __init__(self, model_name,model_weights, training_mean, training_std, training_threshold):
        self.time_steps = 288
        self.training_mean = training_mean
        self.training_std = training_std
        self.training_threshold = training_threshold
        self.model = model_name
        self.weights = model_weights
        
        
    def __call__(self, test_data):
        df_test_value = self.normalize_test(test_data)
        #(test_data - self.training_mean) / self.training_std
        x_test = self.create_sequences(df_test_value.values)
        print('TEST DATA NORMALIZED')
        print("Test input shape: ", x_test.shape)
        
        # Load Model 
        self.loaded_model = self.get_model()
        
        # predict model outputs
        
        model_return_values = self.model_predict(x_test)
        print('PREDICTIONS DONE')
        
        # calculate MAE loss of test dataset
        mae_loss_values = self.get_MAE_loss(x_test)
        anomalies = mae_loss_values > self.training_threshold
        print("Number of anomaly samples: ", np.sum(anomalies))
        print('MAE LOSS of TEST DATA Calculated')
        print('===== PREPARING FINAL OUTPUT =====')
        final_output = test_data.copy()
         
        final_output['Threshold'] = self.training_threshold
        print('Threshold added')
        # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        anomalous_data_indices = []
        for data_idx in range(self.time_steps - 1, len(df_test_value) - self.time_steps + 1):
            if np.all(anomalies[data_idx - self.time_steps + 1 : data_idx]):
                anomalous_data_indices.append(data_idx)
        
        final_output['MAE'] = self.mae_to_orig_mapping(mae_loss_values, len(test_data))
        #print('MAE values added')
        final_output['Anomaly'] = 'NO'
        final_output['Anomaly'].iloc[anomalous_data_indices] = 'YES'
        print('Anomaly added')
        date_string = [str(i.date()) for i in final_output.index]
        final_output['Date'] = date_string
        
        time_string = [str(i.time()) for i in final_output.index]
        final_output['Time'] = time_string
        
        
        return final_output      
        
    # Generated training sequences for use in the model.
    def create_sequences(self, values):
        output = []
        for i in range(len(values) - self.time_steps):
            output.append(values[i : (i + self.time_steps)])
        return np.stack(output)    

    def normalize_test(self,values):
        values -= self.training_mean
        values /= self.training_std
        return values
    
    def model_predict(self,values):
        return self.loaded_model.predict(values)
    
    def get_MAE_loss(self,values):
    # Get test MAE loss.
        x_test_pred = self.loaded_model.predict(values)
        test_mae_loss = np.mean(np.abs(x_test_pred - values), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))
        return test_mae_loss

    def get_model(self):
        json_file = open(self.model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.weights)
        print("Loaded model from disk")
        print(loaded_model.summary())
        return loaded_model

    def mae_to_orig_mapping(self, mae_values, map_length):
        mae_list = list(mae_values)
        print(len(mae_list), map_length,mae_values.shape[0])
        mae_map = np.ones(map_length)
        a = np.arange(self.time_steps, map_length)
        b = np.arange(mae_values.shape[0])
        for ai, bi in zip(a,b):
            if bi == 0:
                mae_map[bi : ai] = (mae_list[bi:ai])
            else:
                mae_map[ai] = (mae_list[bi])
        
        return mae_map.T
        