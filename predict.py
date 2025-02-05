from model_functions import *
import pandas as pd
import joblib

# svm model
# poly dataset

model = joblib.load('model.pkl')
columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32','remove1','remove2']

new_data = pd.DataFrame(pd.read_csv(r"C:\Users\maxol\code\predictive_maintenance\nasa_jet_rul_dataset\test_FD001.txt",delimiter=' ', names=columns))
ingest_data = preprocess(new_data)
ingest_data = transform(ingest_data)

testing = True
if testing: 
    y_true = pd.read_csv(r"C:\Users\maxol\code\predictive_maintenance\nasa_jet_rul_dataset\RUL_FD001.txt", header=None, names=['y_true_regression'])

#for i in range(len(new_data)):
for i in range(10):
    pred = model.predict(ingest_data[i].reshape(1,-1))
    print(f"Input Data: {new_data.iloc[i]}\nPredicted RUL: {int(pred[0])}")
    if testing:
        print(f"Actual RUL: {y_true.iloc[i,0]}")