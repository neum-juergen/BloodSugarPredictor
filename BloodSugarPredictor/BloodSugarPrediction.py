## this is still under construction

import config
import MongoDbConnector
from tensorflow import keras

df = MongoDbConnector.get_entries()
df = MongoDbConnector.transform_df(df)

model = keras.models.load_model(config.ts_folder+'\\best_model_trained_on_val.pb')

#TODO: Put these in config
history_length = 10*24*12  # The history length in 5 minute steps.
step_size = 1  # The sampling rate of the history. Eg. If step_size = 1, then values from every 5 minutes will be in the history.
                #                                       If step size = 10 then values every 50 minutes will be in the history.
target_step = 3  # The time step in the future to predict. Eg. If target_step = 0, then predict the next timestep after the end of the history period.
                  
def transform_ts(dataset,  
                    end_index, 
                    history_length, 
                    step_size):
    assert step_size > 0

    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags]
    if end_index is None:
        end_index = len(dataset)

            
    data_list = []
        
    # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
    
    indices = range(end_index, end_index-history_length, -step_size)
    data = dataset[sorted(indices)]
            
    # append data to the list.
    data_list.append(data)

    df_ts = pd.DataFrame(data=data_list, columns=col_names)
            
    return df_ts


