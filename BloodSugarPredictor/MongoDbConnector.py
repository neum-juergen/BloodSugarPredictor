from pymongo import MongoClient
import config
from datetime import datetime
import pandas as pd

def get_entries(newest_n=None):
    if newest_n is None:
        return sgv_df     
    return sgv_df.tail(newest_n)

def transform_df(df, datestring='dateString'):
    df['DateTime'] = pd.to_datetime(df[datestring])
    print("Sorting...")
    df.sort_values('DateTime', inplace=True, ascending=True)
    return df

def get_treatments(newest_n=None, before=None):
    if before is None:
        before=insulin_df['DateTime'].max()
    if newest_n is not None:
        return insulin_df[insulin_df['DateTime'] <= before].tail(newest_n)
    return insulin_df[insulin_df['DateTime'] <= before]


# MongoDB credentials are stored in a seperate py module named config

print("Connecting to DB")
client = MongoClient(config.mongo_atlas_string)
db = client.get_database(config.db_name)
print ("Connection successful")
records = db.entries.find({"type":"sgv"})
sgv_df =  pd.DataFrame(list(records))
sgv_df = transform_df(sgv_df)

insulin_records = db.treatments.find({"insulin":{"$exists":"true"},"timestamp":{"$exists":"true"}})
insulin_df =  pd.DataFrame(list(insulin_records))
insulin_df = transform_df(insulin_df,'created_at')[['insulin','DateTime']]
print("Number of insulin treatments found:"+str(len(insulin_df)))

