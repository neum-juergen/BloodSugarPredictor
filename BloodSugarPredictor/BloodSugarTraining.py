# This module builds training, validation and test sets. It trains a LSTM model using hyperparameter tuning.


import matplotlib.pyplot as plt
import datetime as dt
from dateutil.parser import parse as dtparser
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import os
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.metrics import mean_squared_error
from TimeSeriesLoader import TimeSeriesLoader
from keras.optimizers import SGD
import numpy as np
from kerastuner.tuners import BayesianOptimization
import MongoDbConnector
import BloodSugarTesting
import config
import pandas as pd
import sys
from pickle import dump
# honor where honor is due! the framework of this model relies on the tutorial posted here: https://towardsdatascience.com/3-steps-to-forecast-time-series-lstm-with-tensorflow-keras-ba88c6f05237

ts_folder = config.ts_folder
ts_val_folder = config.ts_val_folder
ts_test_folder = config.ts_test_folder
filename_format = 'ts_file{}.pkl'





df = MongoDbConnector.get_entries()
df_insulin =  MongoDbConnector.get_treatments()


print("Building datasets...")

test_cutoff_date = df['DateTime'].max() - timedelta(days=14)
val_cutoff_date = test_cutoff_date - timedelta(days=14)

df_test = df[df['DateTime'] > test_cutoff_date]
df_val = df[(df['DateTime'] > val_cutoff_date) & (df['DateTime'] <= test_cutoff_date)]
df_train = df[df['DateTime'] <= val_cutoff_date]

val_cutoff_date = val_cutoff_date + timedelta(minutes=4)

df_insulin = df_insulin[df_insulin['DateTime'] <= val_cutoff_date]
 

#check out the datasets
print('Test dates: {} to {}'.format(df_test['DateTime'].min(), df_test['DateTime'].max()))
print('Validation dates: {} to {}'.format(df_val['DateTime'].min(), df_val['DateTime'].max()))
print('Train dates: {} to {}'.format(df_train['DateTime'].min(), df_train['DateTime'].max()))


sugar_values = df_train['sgv'].values
datetime_values = df_train['DateTime'].values
hour_values = df_train['DateTime'].apply(lambda row: row.hour).values
insulin_values = df_insulin['insulin'].values

history_length = 10*24*12  # The history length in 5 minute steps.
insulin_history_length = 10 # number of last insulin values to use for prediction
step_size = 1  # The sampling rate of the history. Eg. If step_size = 1, then values from every 5 minutes will be in the history.
                #                                       If step size = 10 then values every 50 minutes will be in the history.
target_step = 3  # The time step in the future to predict. Eg. If target_step = 0, then predict the next timestep after the end of the history period.
                  #                                             If target_step = 3 then predict 3 timesteps after the next timestep ((3+1)*5 minutes after the end of history).


# Scaled to work with Neural networks.
scaler = MinMaxScaler(feature_range=(0, 1))
sugar_values_scaled = scaler.fit_transform(sugar_values.reshape(-1, 1)).reshape(-1, )
dump(scaler, open('scaler.pkl', 'wb'))


hour_scaler = MinMaxScaler(feature_range=(0, 1))
hour_values_scaled = hour_scaler.fit_transform(hour_values.reshape(-1, 1)).reshape(-1, )
dump(hour_scaler, open('hour_scaler.pkl', 'wb'))


insulin_scaler = MinMaxScaler(feature_range=(0, 1))
insulin_scaler.fit_transform(insulin_values.reshape(-1, 1)).reshape(-1, )
dump(insulin_scaler, open('insulin_scaler.pkl', 'wb'))


def create_ts_files(hour_data, dataset, dataset_datetimes,
                    start_index, 
                    end_index, 
                    history_length, insulin_history_length,
                    step_size, 
                    target_step, 
                    num_rows_per_file, 
                    data_folder):
    assert step_size > 0
    assert start_index >= 0
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    insulin_time_lags = sorted(range(target_step+1, target_step+insulin_history_length+1, step_size), reverse=True)
    col_names = ['hour']+ [f'x_lag{i}' for i in time_lags] + [f'ins_lag{i}' for i in insulin_time_lags] + ['y']
    start_index = start_index + history_length
    if end_index is None:
        end_index = len(dataset) - target_step
    
    rng = range(start_index, end_index)
    num_rows = len(rng)
    num_files = math.ceil(num_rows/num_rows_per_file)
    
    # for each file.
    print(f'Creating {num_files} files.')
    for i in range(num_files):
        filename = f'{data_folder}/ts_file{i}.pkl'
        
        if i % 10 == 0:
            print(f'{filename}')
            
        # get the start and end indices.
        ind0 = max(i*num_rows_per_file, start_index)
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []
        
        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        for j in range(ind0, ind1):
            indices = range(j-1, j-history_length-1, -step_size)
            data = dataset[sorted(indices) + [j+target_step]]
            data = np.insert(data, 0,hour_data[j-1])
            insulin_values = MongoDbConnector.get_treatments(insulin_history_length,dataset_datetimes[j-1]+timedelta(minutes=4))['insulin'].values
            while len(insulin_values)<insulin_history_length:
                insulin_values=np.insert(insulin_values,0,0)
            insulin_values_scaled = insulin_scaler.transform(insulin_values.reshape(-1, 1)).reshape(-1, )
            data = np.insert(data,len(data)-2,insulin_values_scaled)
            # append data to the list.
            data_list.append(data)
        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)
        
            
    return len(col_names)-1



# The csv creation returns the number of rows and number of features. We need these values below.
num_timesteps = create_ts_files(hour_values_scaled, sugar_values_scaled, datetime_values,
                                start_index=0,
                                end_index=None,
                                history_length=history_length, insulin_history_length=insulin_history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=128*100,
                                data_folder=ts_folder)




tss = TimeSeriesLoader(ts_folder, filename_format)

# Create the Keras model.

def build_model(hp):
    ts_inputs = tf.keras.Input(shape=(num_timesteps, 1))
    x = layers.LSTM(units=hp.Int('units',min_value=10,
                                    max_value=20,
                                    step=1))(ts_inputs)
    x = layers.Dropout(hp.Choice('dropout',
                  values=[0.05, 0.1, 0.2, 0.3, 0.4]))(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

    return model


# create a new keras tuner that overrides run_trial so that we can feed it multiple training datasets in a single trial
class TimeSeriesTuner (BayesianOptimization):
    
    def __init__(self,hypermodel, objective, max_trials, num_initial_points=2, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True, **kwargs):
        self.best_loss = sys.float_info.max
        self.best_epochs = sys.float_info.max
        super().__init__(hypermodel, objective, max_trials, num_initial_points=2, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True, **kwargs)

    def run_trial(self, trial, x, y, val_x, val_y, batch_size, epochs):
        model = self.hypermodel.build(trial.hyperparameters)
        epoch = 0
        current_mse = sys.float_info.max
        best_epoch_mse = sys.float_info.max
        while epoch < epochs and current_mse<=best_epoch_mse:
            epoch+=1
            for i in range(tss.num_chunks()):
                X, Y = tss.get_chunk(i)
        
                model.fit(x=X, y=Y, batch_size=batch_size)
            
            # shuffle the chunks so they're not in the same order next time around.
            tss.shuffle_chunks()
            print("Evaluation...")
            loss = model.evaluate(val_x, val_y)
            print("Evaluation done!")
            current_mse = loss[0]
            if current_mse<best_epoch_mse:
                best_epoch_mse=current_mse
                self.oracle.update_trial(trial.trial_id, {'mse': loss[0]},epoch-1)
                self.save_model(trial.trial_id, model,epoch-1)
            if current_mse<self.best_loss:
               model.save(ts_folder+'\\best_model.pb')
               self.best_loss = loss[0]
               self.best_epochs = epoch
               print("Best mse: "+str(self.best_loss))     
            print("Epoch no:"+str(epoch)+" done!")
        print("Trial with id "+trial.trial_id+" and loss of "+ str(best_epoch_mse) +" is done!")


        
   


# Create the validation CSV like we did before with the training.
sugar_values_val = df_val['sgv'].values
sugar_values_val_scaled = scaler.transform(sugar_values_val.reshape(-1, 1)).reshape(-1, )

datetime_values_val = df_val['DateTime'].values

hour_values_val = df_val['DateTime'].apply(lambda row: row.hour).values
hour_values_val_scaled = hour_scaler.transform(hour_values_val.reshape(-1, 1)).reshape(-1, )


num_timesteps = create_ts_files(hour_values_val_scaled, sugar_values_val_scaled, datetime_values_val,
                                start_index=0,
                                end_index=None,
                                history_length=history_length, insulin_history_length=insulin_history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=128*100,
                                data_folder=ts_val_folder)


# Create the test CSV like we did before with the training.
sugar_values_test = df_test['sgv'].values
sugar_values_test_scaled = scaler.transform(sugar_values_test.reshape(-1, 1)).reshape(-1, )

datetime_values_test = df_test['DateTime'].values

hour_values_test = df_test['DateTime'].apply(lambda row: row.hour).values
hour_values_test_scaled = hour_scaler.transform(hour_values_test.reshape(-1, 1)).reshape(-1, )


num_timesteps = create_ts_files(hour_values_test_scaled, sugar_values_test_scaled, datetime_values_test,
                                start_index=0,
                                end_index=None,
                                history_length=history_length, insulin_history_length=insulin_history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=128*100,
                                data_folder=ts_test_folder)



df_val_ts = pd.read_pickle(ts_val_folder+'\\ts_file0.pkl')
num_records = len(df_val_ts.index)

features = df_val_ts.drop('y', axis=1).values
features_arr = np.array(features)

# reshape for input into LSTM. Batch major format.

features_batchmajor = features_arr.reshape(num_records, -1, 1)

tuner = TimeSeriesTuner(
    build_model,
    objective='mse',
    max_trials=15,
    executions_per_trial=1,
    directory=os.path.normpath('D:/keras_tuning'),
    project_name='kerastuner_bayesian',
    overwrite=True)

x, y = tss.get_chunk(1)

# train in batch sizes
BATCH_SIZE = 128
NUM_EPOCHS = 100

tuner.search(x, y,
             epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
             val_x=features_batchmajor, val_y=df_val_ts['y'].values)


#BloodSugarTesting.test_last_best_model()

def train_model(model,epochs,batch_size):
    epoch = 0
    while epoch < epochs:
            epoch+=1
            for i in range(tss.num_chunks()):
                X, Y = tss.get_chunk(i)
                model.fit(x=X, y=Y, batch_size=batch_size)
            tss.shuffle_chunks()
    return model

hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(hps)
epoch = 0
tss.add_chunk(filename=ts_val_folder+'\\ts_file0.pkl')

model=train_model(model,tuner.best_epochs,BATCH_SIZE)

print("Error on test after training on train+validation:")
df_test_ts = pd.read_pickle(ts_test_folder+'\\ts_file0.pkl')
BloodSugarTesting.test_model_on(model,df_test_ts,scaler)


print("Training on all data")
tss.add_chunk(filename=ts_test_folder+'\\ts_file0.pkl')
model = tuner.hypermodel.build(hps)
model=train_model(model,tuner.best_epochs,BATCH_SIZE)
model.save(ts_folder+'\\best_model_fullytrained.pb')