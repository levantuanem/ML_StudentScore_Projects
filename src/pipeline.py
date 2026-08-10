from src.data.preprocess import preprocess_data
from src.features.build_feature import build_features
from src.models.train import train_models
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model
from src.models.predict import predict


def main():
    # 1. Load + clean data
    data = preprocess_data()

    # 2. Build features + split train/test
    X_train, X_test, y_train, y_test = build_features(data)

    # 3. Train multiple regression models
    results = train_models(X_train, X_test, y_train, y_test)

    # 4. Tune best model
    best_model = tune_model(X_train, y_train)

    # 5. Evaluate tuned model
    evaluate_model(best_model, X_test, y_test)

    # 6. Prediction
    predictions = predict(best_model, X_test)

    return predictions


if __name__ == "__main__":
    main()