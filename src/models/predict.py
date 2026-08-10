import joblib
import pandas as pd
from src.features.build_feature import transform_new_data

def predict_student():
    model = joblib.load(r"D:\AI\_Projects\ML\StudentScore\Projects\models\ridge\best.pkl")
    X = transform_new_data(data)
    prediction = model.predict(X)
    return prediction

if __name__ == "__main__":

    data = pd.read_csv(r"D:\AI_Projects\ML_StudentScore_Projects\data\raw\new_students_for_prediction.csv")
    data = data.rename(columns={
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_level_of_education",
        "test preparation course": "test_preparation_course",
        "reading score": "reading_score",
        "writing score": "writing_score"})
    prediction = predict_student(data)
    data["predicted_math_score"] = prediction
    print("\n======================================")
    print("       STUDENT SCORE PREDICTION")
    print("======================================")
    print(data.to_string(index=False))
