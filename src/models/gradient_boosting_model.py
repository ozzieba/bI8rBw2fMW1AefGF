from sklearn.ensemble import GradientBoostingClassifier
from .trainable_model import TrainableModel

class GradientBoostingModel(TrainableModel):
    def __init__(self):
        super().__init__(GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 10]})
