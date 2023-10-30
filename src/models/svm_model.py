from sklearn.svm import SVC
from .trainable_model import TrainableModel

class SVMModel(TrainableModel):
    def __init__(self):
        super().__init__(SVC(), {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]})
