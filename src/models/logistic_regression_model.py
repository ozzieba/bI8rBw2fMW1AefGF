from sklearn.linear_model import LogisticRegression
from .trainable_model import TrainableModel

class LogisticRegressionModel(TrainableModel):
    def __init__(self):
        super().__init__(LogisticRegression(), {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']})
