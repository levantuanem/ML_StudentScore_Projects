import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'D:\AI_Projects\ML_StudentScore_Projects\data\raw\StudentsPerformance.csv')
print(data.head())
