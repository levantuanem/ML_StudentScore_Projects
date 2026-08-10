from src.data.preprocess import preprocess_data
from src.features.build_feature import build_feature
from src.models.train import train_model
from src.models.tune import tune_ridge
from src.models.evaluate import evaluate_ridge
from src.models.predict import predict_student
from pathlib import Path

def main():

    # 1. Preprocess
    input_path = Path("data/raw/StudentsPerformance.csv")
    output_path = Path("data/processed/StudentsPerformance_clean.csv")
    preprocess_data(input_path, output_path)
    # 2. Feature Engineering
    data = build_feature(output_path)
    # 3. Train
    results = train_model(data)
    # 4. Tune
    best_model = tune_ridge()
    # 5. Evaluate
    evaluate_ridge(best_model, data)

    # 6. Predict
    predict_student(best_model, data)
    return predict_student

if __name__ == "__main__":
    main()