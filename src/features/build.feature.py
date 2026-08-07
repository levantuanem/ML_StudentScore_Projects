import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import data

def build_feature():
    data = pd.read_csv(r'D:\AI_Projects\ML_StudentScore_Projects\models\StudentsPerformance_clean.csv', sep=';')
    x = data.drop(['math score', 'reading score', 'writing score'], axis=1)
    y = data[['math score', 'reading score', 'writing score']]
    return x, y

categorical_cols = ["parental level of education", "lunch", "test preparation course"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
joblib.dump(encoded_df, "models/encoded.pkl")


if __name__ == "__main__":
    x, y = build_feature()
    print(x.head())
    print(y.head())