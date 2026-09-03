import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_TEST_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data" / "X_test.csv"
DATA_TRAIN_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data" / "X_train.csv"

OUTPUT_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data"

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


def save_dataframe(df, output_dir, save_name):
    output_filepath = os.path.join(output_dir, f'{save_name}.csv')
    df.to_csv(output_filepath, index=False)

def normalize_df(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def main():
    test_data = import_dataset(DATA_TEST_PATH)
    train_data = import_dataset(DATA_TRAIN_PATH)

    scaled_test_data = normalize_df(test_data)
    scaled_train_data = normalize_df(train_data)

    save_dataframe(scaled_test_data, OUTPUT_PATH, "X_test_scaled")
    save_dataframe(scaled_train_data, OUTPUT_PATH, "X_train_scaled")

if __name__ == "__main__":
    main()