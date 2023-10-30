from sklearn.ensemble import RandomForestClassifier
from .trainable_model import TrainableModel

class RandomForestModel(TrainableModel):
    def __init__(self):
        super().__init__(RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]})
