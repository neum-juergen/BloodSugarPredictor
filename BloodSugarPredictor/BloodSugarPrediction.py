from datetime import timedelta
import config
import MongoDbConnector
from tensorflow import keras
from pickle import load
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#TODO: Put these in config
history_length = 10*24*12  # The history length in 5 minute steps.
insulin_history_length = 10
step_size = 1  # The sampling rate of the history. Eg. If step_size = 1, then values from every 5 minutes will be in the history.
                #                                       If step size = 10 then values every 50 minutes will be in the history.
target_step = 3  # The time step in the future to predict. Eg. If target_step = 0, then predict the next timestep after the end of the history period.


df = MongoDbConnector.get_entries(history_length)


model = keras.models.load_model(config.ts_folder+'\\best_model_fullytrained.pb')
scaler = load(open('scaler.pkl', 'rb'))

sugar_values = df['sgv'].values
sugar_values_scaled = scaler.fit_transform(sugar_values.reshape(-1, 1)).reshape(-1, )

hour_scaler = load(open('hour_scaler.pkl', 'rb'))
hour_values = df['DateTime'].apply(lambda row: row.hour).values
hour_values_scaled = hour_scaler.fit_transform(hour_values.reshape(-1, 1)).reshape(-1, )
datetime_values = df['DateTime'].values
insulin_scaler = load(open('insulin_scaler.pkl', 'rb'))

def transform_ts(hour_data, dataset, dataset_datetimes, 
                    end_index, 
                    history_length, insulin_history_length,
                    step_size):
    assert step_size > 0

    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    insulin_time_lags = sorted(range(target_step+1, target_step+insulin_history_length+1, step_size), reverse=True)
    col_names = ['hour']+ [f'x_lag{i}' for i in time_lags] + [f'ins_lag{i}' for i in insulin_time_lags] 
    if end_index is None:
        end_index = len(dataset)

            
    data_list = []
         
    indices = range(end_index-1, end_index-history_length-1, -step_size)
    data = dataset[sorted(indices)]
    data = np.insert(data, 0,hour_data[end_index-1])
    insulin_values = MongoDbConnector.get_treatments(insulin_history_length,dataset_datetimes[end_index-1]+timedelta(minutes=4))['insulin'].values
    while len(insulin_values)<insulin_history_length:
        insulin_values=np.insert(insulin_values,0,0)
    insulin_values_scaled = insulin_scaler.transform(insulin_values.reshape(-1, 1)).reshape(-1, )
    data = np.insert(data,len(data)-2,insulin_values_scaled)
    # append data to the list.
    data_list.append(data)

    df_ts = pd.DataFrame(data=data_list, columns=col_names)
            
    return df_ts


df_transformed = transform_ts(hour_values_scaled,sugar_values_scaled,datetime_values,end_index=None,history_length=history_length, insulin_history_length=insulin_history_length,step_size=step_size)
features_arr = np.array(df_transformed)

# reshape for input into LSTM. Batch major format.
num_records = len(df_transformed.index)
features_batchmajor = features_arr.reshape(num_records, -1, 1)
y_pred = model.predict(features_batchmajor).reshape(-1, )
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1 ,)

print("Latest value from "+df.tail(1).iloc[0]['dateString'])
print("Prediction: "+str(y_pred[0]))
