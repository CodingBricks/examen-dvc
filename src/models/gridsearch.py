from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pandas as pd
import os 
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data"
PARAM_SAVE_PATH = SCRIPT_DIR / ".." / ".." / "models" / "best_params.pkl"

def load_data(csv_name):
    load_path = os.path.join(DATA_PATH, f'{csv_name}.csv')
    return pd.read_csv(load_path)

def save_params(params):
    with open(PARAM_SAVE_PATH, "wb") as f:
        pickle.dump(params, f)


def main():

    X_train_scaled = load_data("X_train_scaled")
    X_test_scaled = load_data("X_test_scaled")
    y_train = load_data("y_train")
    y_test = load_data("y_test")

    xgb = XGBRegressor(random_state=42, n_jobs=1) # set n_jobs=1 to prevent collision with GridSearchCV n_jobs=-1

    param_grid = {'n_estimators': [50, 100, 150, 200],
                  'max_depth': [5,10,15,20],
                  'learning_rate': [0.01, 0.1, 0.2],
                  'subsample': [0.8, 1.0]}

    grid_search = GridSearchCV(estimator=xgb,
                               param_grid=param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1,
                               verbose=1)

    print("Searching for best XGBoost parameters ...")
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_

    print(f"Best Parameters: {best_params}")
    print(f"Best CV Mean Squared Error: {-grid_search.best_score_:.4f}")

    save_params(best_params)


if __name__ == "__main__":
    main()