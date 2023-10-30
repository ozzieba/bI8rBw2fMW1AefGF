import os
import importlib
import inspect
from .data_loader import load_data, get_repo_root
from .trainable_model import TrainableModel

def train_all_models():
    X_train, X_test, y_train, y_test = load_data()
    repo_root = get_repo_root()

    for filename in os.listdir(f'{repo_root}/src/models'):
        if filename.endswith('_model.py'):
            module_name = filename[:-3]
            module = importlib.import_module('src.models.' + module_name)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, TrainableModel) and obj is not TrainableModel:
                    model = obj()
                    model.grid_search(X_train, y_train)
                    model.save(f'{repo_root}/models/{module_name}_model.pkl')

train_all_models()
