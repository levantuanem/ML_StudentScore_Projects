import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from src.features.build_feature import build_feature

def tune_ridge():
    # Lấy data
    x_train, x_test, y_train, y_test = build_feature()
    # Model chọn ra từ bước train
    model_ridge = Ridge()
    #  Hyperparameter cần tune
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "fit_intercept": [True, False]}
    # Grid search
    grid_search = GridSearchCV(estimator=model_ridge, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1)
    # Tune trên training set
    grid_search.fit(x_train, y_train)
    print("Best parameter: ")
    print(grid_search.best_params_)
    print("Best CV r2: ")
    print(grid_search.best_score_)
    # Model tốt nhất
    best_model = grid_search.best_estimator_
    # Lưu model
    joblib.dump(best_model, r"D:\AI_Projects\ML_StudentScore_Projects\models/ridge_best.pkl")

    return best_model

if __name__ == "__main__":
    tune_ridge()