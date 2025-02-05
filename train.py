from model_functions import *
import pandas as pd
import joblib

# svm model
# poly dataset

hyperparameters = {'kernel':'poly', 'epsilon':.89, 'degree':1, 'C':20}
model = SVR(**hyperparameters)
columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32','remove1','remove2']

train_data = pd.read_csv(r"C:\Users\maxol\code\predictive_maintenance\nasa_jet_rul_dataset\train_FD001.txt",delimiter=' ', names=columns)
train_data = generateRUL(train_data,0)
y = train_data['RUL']

train_data = preprocess(train_data)
train_data = transform(train_data)

model.fit(train_data, y)

joblib.dump(model, 'model.pkl')