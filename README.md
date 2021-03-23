# BloodSugarPredictor
This is a hobby project that aims to help a friend of mine with diabetes type 1 to make use of their CGM data for prediction of future blood sugar values.

Currently, I try to predict blood sugar values 20 minutes into the future using a LSTM model. Current best mean squared error on the test set for my friend's sugar values is about 18 on a scale which usually runs from 39 to 400.

# Requirements

I assume that you have access to blood sugar values collected by a Continous Glucose Monitoring System (CGM). Specifically, data should be stored in a MongoDB with a collection called 'entries'. In the case of my friend, she uses a Dexcom sensor to collect sugar values every 5 minutes, which is then stored to MongoDB using xDrip and Nightscout (for more information please refer to http://www.nightscout.info/).

# Installation

Easiest way to install this would be to clone the repository using Visual Studio. Then, look into config_mockup.py and make adjustments with your own parameters that you need to get from your own database. Do not forget to change the name to config.py.

First run BloodSugarTraining.py until at least one trial is finished. As a general rule of thumb, the longer you let it run, the more accurate the predictions. Also, you can keep it running and still use the current best solution using BloodSugarPrediction.py. If at least one trial is finished, BloodSugarPrediction.py outputs the time of the latest reading and its prediction for the blood sugar value 20 minutes later.
