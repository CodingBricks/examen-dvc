from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import json


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / ".." / ".." / "models" / "model.pkl"
DATA_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data"
METRICS_SAVE_PATH = SCRIPT_DIR / ".." / ".." / "metrics" / "scores.json"

def load_data(csv_name):
    load_path = os.path.join(DATA_PATH, f'{csv_name}.csv')
    return pd.read_csv(load_path)

def load_model(model_path):
    with open(model_path, "rb") as f:
        loaded_xgb_model = pickle.load(f)
    return loaded_xgb_model

def evaluate_model():
    xgb_model = load_model(MODEL_PATH)

    # save predictions as csv in data folder

    X_test_scaled = load_data("X_test_scaled")
    y_test = load_data("y_test")

    y_pred = xgb_model.predict(X_test_scaled)

    output_filepath = os.path.join(DATA_PATH, 'predictions.csv')
    pred_df = pd.DataFrame(y_pred, columns=["silica_concentrate"])
    pred_df.to_csv(output_filepath, index=False)

    # evaluate model on MSE and R2 and save scores as scores.json in the metrics folder

    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    scores_dict = {
        "mean_squared_error": mse,
        "r2_score": r2
    }

    with open(METRICS_SAVE_PATH, "w") as jf:
        json.dump(scores_dict, jf, indent=4)


if __name__ == "__main__":
    evaluate_model()