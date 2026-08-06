import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as StandardScaler

data = pd.read_csv(r'D:\AI_Projects\ML_StudentScore_Projects\data\raw\StudentsPerformance.csv')
print(data.info())
print("Original lenght of Dataframe: {}".format(len(data)))
data.dropna(inplace= True)
print("Lenght of Dataframe afer removing NAN values: {}".format(len(data)))