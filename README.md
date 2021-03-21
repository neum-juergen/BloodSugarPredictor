# BloodSugarPredictor
This is a hobby project that aims to help a friend of mine with diabetes type 1 to make use of their CGM data for prediction of future blood sugar values.

Currently, I try to predict blood sugar values 20 minutes into the future using a LSTM model.

# Requirements

I assume that you have access to blood sugar values collected by a Continous Glucose Monitoring System (CGM). Specifically, data should be stored in a MongoDB with a collection called 'entries'. In the case of my friend, she uses a Dexcom sensor to collect sugar values every 5 minutes, which is then stored to MongoDB using xDrip and Nightscout (for more information please refer to http://www.nightscout.info/).

# Installation

Easiest way to install this would be to pull the repository using Visual Studio. Then, create a new Python module called config with two parameters that you need to get from your own database.

mongo_atlas_string = "Enter string from MongoDB Atlas here"
db_name= 'Enter DB name here'
