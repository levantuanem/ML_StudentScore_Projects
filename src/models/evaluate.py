import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from src.features.build_feature import build_feature
def evaluate_ridge():
    x_train, x_test, y_train, y_test = build_feature()
    model = joblib.load(r"models/ridge\_best.pkl")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print("\n======================================")
    print("       RIDGE MODEL EVALUATION")
    print("======================================")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
if __name__ == "__main__":
    evaluate_ridge()