from sklearn.neighbors import KNeighborsClassifier
from .trainable_model import TrainableModel

class KNNModel(TrainableModel):
    def __init__(self):
        super().__init__(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
