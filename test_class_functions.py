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
    
    def __init__(self, training_mean, training_std, training_threshold):
        self.time_steps = 288
        self.training_mean = training_mean
        self.training_std = training_std
        self.training_threshold = training_threshold
        
        
    def __call__(self, test_data):
        df_test_value = self.normalize_test(test_data)
        #(test_data - self.training_mean) / self.training_std
        x_test = self.create_sequences(df_test_value.values)
        print("Test input shape: ", x_test.shape)
        
        # predict model outputs
        model_return_values = self.model_predict(x_test)
        
        # calculate MAE loss of test dataset
        mae_loss_values = self.get_MAE_loss(model_return_values)
        
              
        
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
        return self.model.predict(values)
    
    def get_MAE_loss(self,values):
    # Get test MAE loss.
        x_test_pred = model.predict(values)
        test_mae_loss = np.mean(np.abs(x_test_pred - values), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))
        return test_mae_loss



