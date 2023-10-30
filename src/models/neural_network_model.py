from sklearn.neural_network import MLPClassifier
from .trainable_model import TrainableModel

class NeuralNetworkModel(TrainableModel):
    def __init__(self):
        super().__init__(MLPClassifier(max_iter=1000), {'hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 20)], 'alpha': [0.0001, 0.001, 0.01]})
