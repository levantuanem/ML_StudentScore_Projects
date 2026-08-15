from src.features.build_feature import build_feature
import pandas as pd
import time
from sklearn.linear_model import(LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

def train_model():
    # =========================
    # BUILD FEATURES
    # =========================
    x_train, x_test, y_train, y_test = build_feature()

    # =========================
    # MODELS
    # =========================
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

    # =========================
    # TRAIN & EVALUATE
    # =========================
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # -------------------------
        # Training time
        # -------------------------
        start_time = time.time()
        model.fit(x_train, y_train)
        training_time = time.time() - start_time

        # -------------------------
        # Test prediction
        # -------------------------
        y_pred = model.predict(x_test)

        # -------------------------
        # Test metrics
        # -------------------------
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        # -------------------------
        # Cross Validation
        # -------------------------
        cv_scores = cross_val_score(model, x_train, y_train,
            cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "CV RMSE": cv_rmse,
            "CV STD": cv_std,
            "Training Time": training_time})

        print(f"MAE         : {mae:.4f}")
        print(f"MSE         : {mse:.4f}")
        print(f"RMSE        : {rmse:.4f}")
        print(f"R2          : {r2:.4f}")
        print(f"CV RMSE     : {cv_rmse:.4f}")
        print(f"CV STD      : {cv_std:.4f}")
        print(f"Training Time: {training_time:.2f}s")

    # =========================
    #  Compare models
    # =========================
    results_df = pd.DataFrame(results)
    # RMSE thấp hơn = tốt hơn
    results_df = results_df.sort_values(by="CV RMSE", ascending=True)
    
    print("\n======================================")
    print("         REGRESSION COMPARISON")
    print("======================================")
    print(results_df.to_string(index=False))
    return results_df

if __name__  == "__main__":
    train_model()