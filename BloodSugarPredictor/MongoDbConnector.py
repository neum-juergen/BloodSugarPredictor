from pymongo import MongoClient
import config

import pandas as pd

def get_entries(newest_n=None):
    print("Connecting to DB")
    # MongoDB credentials are stored in a seperate py module named config
    client = MongoClient(config.mongo_atlas_string)
    db = client.get_database(config.db_name)
    records = db.entries.find({"type":"sgv"})
    print ("Connection successful")
    print("Building DataFrame")
    if newest_n is None:
        df =  pd.DataFrame(list(records))
    else:
        df = pd.DataFrame(list(records.skip(records.count() - newest_n)))
    return df

def transform_df(df):
    df['DateTime'] = pd.to_datetime(df['dateString'])
    print("Building DataFrame successful")

    print("Sorting...")
    df.sort_values('DateTime', inplace=True, ascending=True)
    return df
