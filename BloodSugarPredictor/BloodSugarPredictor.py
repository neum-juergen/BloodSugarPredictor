from pymongo import MongoClient
import pandas as pd
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

import config

# honor where honor is due! the framework of this model relies on the tutorial posted here: https://towardsdatascience.com/3-steps-to-forecast-time-series-lstm-with-tensorflow-keras-ba88c6f05237

ts_folder = 'D:\BloodSugarPredictor\BloodSugarTrainData'
ts_val_folder = 'D:\BloodSugarPredictor\BloodSugarValidationData'
filename_format = 'ts_file{}.pkl'







print("Connecting to DB")
# MongoDB credentials are stored in a seperate py module named config
client = MongoClient(config.mongo_atlas_string)
db = client.get_database(config.db_name)
records = db.entries
print ("Connection successful")
print("Building DataFrame")
df =  pd.DataFrame(list(records.find({"type":"sgv"})))
#df['DateTime'] = df.apply(lambda row: pd.to_datetime(row.dateString), axis=1)
df['DateTime'] = pd.to_datetime(df['dateString'])
print("Building DataFrame successful")


print("Plotting recent data...")
plot = df[df['DateTime']>pd.to_datetime("2021-03-17T00:00:18.992+0200")].plot(x='DateTime',y='sgv')
#plt.show()

print("Sorting...")
df.sort_values('DateTime', inplace=True, ascending=True)

print("Building datasets...")

test_cutoff_date = df['DateTime'].max() - timedelta(days=7)
val_cutoff_date = test_cutoff_date - timedelta(days=14)

df_test = df[df['DateTime'] > test_cutoff_date]
df_val = df[(df['DateTime'] > val_cutoff_date) & (df['DateTime'] <= test_cutoff_date)]
df_train = df[df['DateTime'] <= val_cutoff_date]

#check out the datasets
print('Test dates: {} to {}'.format(df_test['DateTime'].min(), df_test['DateTime'].max()))
print('Validation dates: {} to {}'.format(df_val['DateTime'].min(), df_val['DateTime'].max()))
print('Train dates: {} to {}'.format(df_train['DateTime'].min(), df_train['DateTime'].max()))


print(df_train['sgv'].isnull().values.sum())

sugar_values = df_train['sgv'].values

# Scaled to work with Neural networks.
scaler = MinMaxScaler(feature_range=(0, 1))
sugar_values_scaled = scaler.fit_transform(sugar_values.reshape(-1, 1)).reshape(-1, )

history_length = 10*24*12  # The history length in 5 minute steps.
step_size = 1  # The sampling rate of the history. Eg. If step_size = 1, then values from every 5 minutes will be in the history.
                #                                       If step size = 10 then values every 50 minutes will be in the history.
target_step = 3  # The time step in the future to predict. Eg. If target_step = 0, then predict the next timestep after the end of the history period.
                  #                                             If target_step = 3 then predict 3 timesteps after the next timestep ((3+1)*5 minutes after the end of history).


def create_ts_files(dataset, 
                    start_index, 
                    end_index, 
                    history_length, 
                    step_size, 
                    target_step, 
                    num_rows_per_file, 
                    data_folder):
    assert step_size > 0
    assert start_index >= 0
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags] + ['y']
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
        ind0 = i*num_rows_per_file
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []
        
        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        for j in range(ind0, ind1):
            indices = range(j-1, j-history_length-1, -step_size)
            data = dataset[sorted(indices) + [j+target_step]]
            
            # append data to the list.
            data_list.append(data)

        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)
            
    return len(col_names)-1



# The csv creation returns the number of rows and number of features. We need these values below.
num_timesteps = create_ts_files(sugar_values_scaled,
                                start_index=0,
                                end_index=None,
                                history_length=history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=128*100,
                                data_folder=ts_folder)




tss = TimeSeriesLoader(ts_folder, filename_format)

# Create the Keras model.

def build_model(hp):
    ts_inputs = tf.keras.Input(shape=(num_timesteps, 1))
    x = layers.LSTM(units=hp.Int('units',min_value=10,
                                    max_value=512,
                                    step=32))(ts_inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(hp.Int('units',min_value=1,
                                    max_value=10,
                                    step=1), activation='linear')(x)

    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)


    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

    return model


# create a new keras tuner that overrides run_trial so that we can feed it multiple training datasets in a single trial
class TimeSeriesTuner (BayesianOptimization):
    def __init__(self,hypermodel, objective, max_trials, num_initial_points=2, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True, **kwargs):
        super().__init__(hypermodel, objective, max_trials, num_initial_points=2, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True, **kwargs)
    def run_trial(self, trial, x, y, val_x, val_y, batch_size, epochs):
        model = self.hypermodel.build(trial.hyperparameters)
        for epoch in range(epochs):
          
            for i in range(tss.num_chunks()):
                X, Y = tss.get_chunk(i)
        
                model.fit(x=X, y=Y, batch_size=batch_size)
        # shuffle the chunks so they're not in the same order next time around.
        tss.shuffle_chunks()
        loss = model.evaluate(val_x, val_y)
        self.oracle.update_trial(trial.trial_id, {'loss': loss})
        self.save_model(trial.trial_id, model)


        
   


# Create the validation CSV like we did before with the training.
sugar_values_val = df_val['sgv'].values
sugar_values_val_scaled = scaler.transform(sugar_values_val.reshape(-1, 1)).reshape(-1, )


# The csv creation returns the number of rows and number of features. We need these values below.
num_timesteps = create_ts_files(sugar_values_val_scaled,
                                start_index=0,
                                end_index=None,
                                history_length=history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=128*100,
                                data_folder=ts_val_folder)

df_val_ts = pd.read_pickle(ts_val_folder+'\\ts_file0.pkl')


features = df_val_ts.drop('y', axis=1).values
features_arr = np.array(features)

# reshape for input into LSTM. Batch major format.
num_records = len(df_val_ts.index)
features_batchmajor = features_arr.reshape(num_records, -1, 1)

tuner = TimeSeriesTuner(
    build_model,
    objective='mse',
    max_trials=3,
    executions_per_trial=1,
    directory=os.path.normpath('D:/keras_tuning'),
    project_name='kerastuner_bayesian',
    overwrite=True)

x, y = tss.get_chunk(1)

# train in batch sizes of 128.
BATCH_SIZE = 128
NUM_EPOCHS = 2

tuner.search(x, y,
             epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
             val_x=features_batchmajor, val_y=df_val_ts['y'].values)

print(tuner.results_summary())

#y_pred = model.predict(features_batchmajor).reshape(-1, )
#y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1 ,)

#y_act = df_val_ts['y'].values
#y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1 ,)

#print('validation mean squared error: {}'.format(mean_squared_error(y_act, y_pred)))