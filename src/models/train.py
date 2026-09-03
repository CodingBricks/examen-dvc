import pickle
from pathlib import Path
from xgboost import XGBRegressor
import pandas as pd
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PARAM_PATH = SCRIPT_DIR / ".." / ".." / "models" / "best_params.pkl"
DATA_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data"
OUTPUT_SAVE_PATH = SCRIPT_DIR / ".." / ".." / "models" / "model.pkl" 


def load_data(csv_name):
    load_path = os.path.join(DATA_PATH, f'{csv_name}.csv')
    return pd.read_csv(load_path)


def load_params(param_path):
    with open(param_path, "rb") as f:
        loaded_params = pickle.load(f)

    return loaded_params

def train_and_save_model():

    X_train_scaled = load_data("X_train_scaled")
    y_train = load_data("y_train")

    best_params = load_params(PARAM_PATH)
    xgb_model = XGBRegressor(**best_params, random_state=42)

    xgb_model.fit(X_train_scaled, y_train)

    with open(OUTPUT_SAVE_PATH, "wb") as f:
        pickle.dump(xgb_model, f)


if __name__ == "__main__":
    train_and_save_model()