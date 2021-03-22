import MongoDbConnector
from tensorflow import keras
import config
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pickle import load



def test_last_best_model():

    scaler = load(open('scaler.pkl', 'rb'))
    if os.path.exists(config.ts_folder+'\\best_model.pb'):
        model = keras.models.load_model(config.ts_folder+'\\best_model.pb')
        df_val_ts = pd.read_pickle(config.ts_val_folder+'\\ts_file0.pkl')
        df_test_ts = pd.read_pickle(config.ts_test_folder+'\\ts_file0.pkl')

        print("MSE on validation set:")
        test_model_on(model, df_val_ts, scaler)

        print("MSE on test set:")
        test_model_on(model, df_test_ts, scaler)

        print("MSE on test set after training on validation:")
        model = train_model_on(model,df_val_ts)
        test_model_on(model,df_test_ts, scaler)

        model.save(config.ts_folder+'\\best_model_trained_on_val.pb')

def test_model_on(best_model, df, scaler):
    features = df.drop('y', axis=1).values
    features_arr = np.array(features)

    # reshape for input into LSTM. Batch major format.
    num_records = len(df.index)
    features_batchmajor = features_arr.reshape(num_records, -1, 1)


    y_pred = best_model.predict(features_batchmajor).reshape(-1, )
    y_act = df['y'].values
    mse = mean_squared_error(y_act, y_pred)
    print('mean squared error on scaled values: {}'.format(mse))

    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1 ,)
    y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1 ,)
    print('mean squared error: {}'.format(mean_squared_error(y_act, y_pred)))
    return mse

    

def train_model_on(best_model, df):
    features = df.drop('y', axis=1).values
    features_arr = np.array(features)

    # reshape for input into LSTM. Batch major format.
    num_records = len(df.index)
    features_batchmajor = features_arr.reshape(num_records, -1, 1)
    best_model.fit(features_batchmajor,df['y'].values)
    return best_model


