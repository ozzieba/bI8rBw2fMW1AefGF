from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import joblib

class TrainableModel:
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid

    def grid_search(self, X_train, y_train):
        scorer = make_scorer(f1_score)
        self.grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=2)
        self.grid_search.fit(X_train, y_train)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
