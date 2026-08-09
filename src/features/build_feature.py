import pandas as pd
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, OrdinalEncoder)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_feature():
    data = pd.read_csv(r"D:\AI_Projects\ML_StudentScore_Projects\data\processed\StudentsPerformance_processed.csv", sep=',')
    target = "math_score"
    x = data.drop(columns= [target])
    y = data[target]

    # Khai báo các nhóm columns
    numeric_features = ["reading_score", "writing_score"]
    ordinal_features = ["parental_level_of_education"]
    normal_features = ["gender", "race_ethnicity", "lunch", "test_preparation_course"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  random_state= 42)

    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
        ])

    # Education pipeline
    education_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
    "master's degree"]
    education_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", fill_value= "unknown")),
    ("encoder", OrdinalEncoder(categories=[education_order]))
    ])

    # Onehot pipeline
    categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", fill_value= "unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Ghép bằng ColumnTransformer
    preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("ordinal", education_pipeline, ordinal_features),
        ("normal", categorical_pipeline, normal_features)
    ])

    # Fit
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    # Save
    joblib.dump(preprocessor,r"D:\AI_Projects\ML_StudentScore_Projects\models\preprocessor.pkl")
    print("Preprocessor saved successfully!")
    # 
    return x_train, x_test, y_train, y_test

def transform_new_data(data):
    preprocessor = joblib.load(r"D:\AI\_Projects**\M**L\_StudentScore\_Projects**\m**odels**\p**reprocessor.pkl")
    X = preprocessor.transform(data)
    return X

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = build_feature()