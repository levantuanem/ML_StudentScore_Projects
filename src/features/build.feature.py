import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
def build_feature():
    data = pd.read_csv(r"D:\AI_Projects\ML_StudentScore_Projects\data\processed\StudentsPerformance_processed.csv", sep=',')
    print(data.columns.tolist())
    target = "math_score"
    x = data.drop(columns= [target])
    y = data[target]

    numeric_features = ["gender", "reading_score", "writing_score"]
    categorical_features = ["race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  random_state= 42)
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    x_train[['reading_score', 'writing_score']] = (numeric_transformer.fit_transform(x_train[['reading_score', 'writing_score']]))
    x_test[['reading_score', 'writing_score']] = (numeric_transformer.transform(x_test[['reading_score', 'writing_score']]))

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = build_feature()