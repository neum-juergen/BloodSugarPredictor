# Taken from here: https://towardsdatascience.com/3-steps-to-forecast-time-series-lstm-with-tensorflow-keras-ba88c6f05237
import pandas as pd
import os
import numpy as np

filename_format = 'ts_file{}.pkl'


class TimeSeriesLoader(object):
    def __init__(self, ts_folder, filename_format):
        self.ts_folder = ts_folder
        self.filenames = []
        # find the number of files.
        i = 0
        file_found = True
        while file_found:
            filename = self.ts_folder + '/' + filename_format.format(i)
            file_found = os.path.exists(filename)
            if file_found:
                self.filenames.append(filename)
                i += 1
                
        self.num_files = i
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()
        
    def num_chunks(self):
        return self.num_files
    
    def get_chunk(self, idx):
        assert (idx >= 0) and (idx < self.num_files)
        
        ind = self.files_indices[idx]
        filename = self.filenames[ind]
        df_ts = pd.read_pickle(filename)
        num_records = len(df_ts.index)
        
        features = df_ts.drop('y', axis=1).values
        target = df_ts['y'].values
        
        # reshape for input into LSTM. Batch major format.
        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target
    
    # this shuffles the order the chunks will be outputted from get_chunk.
    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)

    def add_chunk(self,filename):
        file_found = os.path.exists(filename)  
        if file_found:
            self.filenames.append(filename)
            self.num_files += 1
            self.files_indices = np.append(self.files_indices,len(self.files_indices)-1)