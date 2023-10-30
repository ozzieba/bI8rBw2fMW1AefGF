from sklearn.tree import DecisionTreeClassifier
from .trainable_model import TrainableModel

class DecisionTreeModel(TrainableModel):
    def __init__(self):
        super().__init__(DecisionTreeClassifier(), {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]})
