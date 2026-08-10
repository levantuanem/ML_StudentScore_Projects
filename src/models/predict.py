import joblib
import pandas as pd

from src.features.build_feature import transform_new_data


MODEL_PATH = r"D:\AI_Projects\ML_StudentScore_Projects\models\ridge_best.pkl"
DATA_PATH = r"D:\AI_Projects\ML_StudentScore_Projects\data\raw\new_students_for_prediction.csv"


def predict_student():
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns={
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_level_of_education",
        "test preparation course": "test_preparation_course",
        "reading score": "reading_score",
        "writing score": "writing_score"})
    model = joblib.load(MODEL_PATH)
    X = transform_new_data(data)
    prediction = model.predict(X)
    data["predicted_math_score"] = prediction
    print("\n======================================")
    print("       STUDENT SCORE PREDICTION")
    print("======================================")
    print(data.to_string(index=False))

    return prediction

if __name__ == "__main__":
    predict_student()