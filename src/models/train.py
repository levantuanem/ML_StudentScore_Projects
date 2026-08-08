import pandas as pd
import joblib
from src.features.build_feature import build_feature
from sklearn.linear_model import(LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

def train_model():
    x_train, x_test, y_train, y_test = build_feature()

    models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)}

    results =[]
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2})

        print(f"MAE : {mae:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")

    # =========================
    # 4. Compare models
    # =========================
    results_df = pd.DataFrame(results)
    # RMSE thấp hơn = tốt hơn
    results_df = results_df.sort_values(by="RMSE", ascending=True)
    
    print("\n======================================")
    print("         REGRESSION COMPARISON")
    print("======================================")
    print(results_df.to_string(index=False))
    return results_df
if __name__  == "__main__":
    train_model()