import pandas as pd
import os

from sklearn.model_selection import train_test_split
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / ".." / ".." / "data" / "raw_data" / "raw.csv"
OUTPUT_PATH = SCRIPT_DIR / ".." / ".." / "data" / "processed_data"

#csv_path = "../../data/raw_data/raw.csv"
#output_path = "../../data/processed_data"

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)


def process_data(csv_path, output_path):

    df = import_dataset(csv_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    save_dataframes(X_train, X_test, y_train, y_test, output_path)


if __name__ == "__main__":
    process_data(DATA_PATH, OUTPUT_PATH)