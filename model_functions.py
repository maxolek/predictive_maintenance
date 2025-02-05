import numpy as np
import pandas as pd

# analysis
from scipy.optimize import curve_fit

# pre-processing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA 
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

# modeling
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier

# hyperparameter tuning
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import ParameterGrid, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import make_scorer
#from hyperopt import hp, tpe, fmin, STATUS_OK, Trials 

# evaluation
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error, recall_score, precision_score, roc_auc_score
from sklearn import tree

# feature importance
from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector, RFE
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import ExhaustiveFeatureSelector




########    ETL - FUNCTIONS   ########
# poly dataset

columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
            'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32','remove1','remove2']
constant_columns = ['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'] # columns that have little/no variability and thus serve little/no purpose
features = [_ for _ in columns if _ not in constant_columns+['P15','RUL','risk','remove1','remove2','unit_number','time_in_cycles']]

def generateRUL(train, factor=0):
    df = train.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']

    df = df.merge(fd_RUL, how='left', on=['unit_number'])
    df['RUL'] = df['max'] - df['time_in_cycles']

    df.drop(columns=['max'],inplace=True)
    return df[df['time_in_cycles'] > factor]

def transform(data, degree=3):
    df = data.copy()
    poly = PolynomialFeatures(degree=degree,include_bias=False)
    transformed = poly.fit_transform(df)
    return transformed

def preprocess(data):
    df = data.copy()
    df = df[features]
    scaler = StandardScaler()
    transformed = scaler.fit_transform(df)
    return transformed
