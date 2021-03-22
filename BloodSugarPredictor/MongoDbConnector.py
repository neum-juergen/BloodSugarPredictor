from pymongo import MongoClient
import config

import pandas as pd

def get_entries():
    print("Connecting to DB")
    # MongoDB credentials are stored in a seperate py module named config
    client = MongoClient(config.mongo_atlas_string)
    db = client.get_database(config.db_name)
    records = db.entries
    print ("Connection successful")
    print("Building DataFrame")
    df =  pd.DataFrame(list(records.find({"type":"sgv"})))
    return df

def transform_df(df):
    df['DateTime'] = pd.to_datetime(df['dateString'])
    print("Building DataFrame successful")

    print("Sorting...")
    df.sort_values('DateTime', inplace=True, ascending=True)
    return df
