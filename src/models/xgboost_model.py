from xgboost import XGBClassifier
from .trainable_model import TrainableModel

class XGBoostModel(TrainableModel):
    def __init__(self):
        super().__init__(XGBClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 10], 'reg_alpha': [0, 0.5, 1], 'reg_lambda': [1, 1.5, 2]})
