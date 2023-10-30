import os
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split


def get_repo_root():
    """Return the absolute path of the repository root."""
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')


def load_data():
    repo_root = get_repo_root()
    data_path = os.path.join(repo_root, 'data/processed/customer_data.csv')
    data = pd.read_csv(data_path)
    X = data.drop('Y', axis=1)
    y = data['Y']
    return train_test_split(X, y, test_size=0.2, random_state=42)
