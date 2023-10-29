import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib

# Load the data
data = pd.read_csv('data/raw/customer_data.csv')

# Create a new feature
data['X_min'] = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].min(axis=1)

# Split the data
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'svm': SVC(),
    'decision_tree': DecisionTreeClassifier(),
    'knn': KNeighborsClassifier(),
    'neural_network': MLPClassifier(max_iter=1000),
    'adaboost': AdaBoostClassifier(),
    'xgboost': XGBClassifier()
}

# Define the parameter grids
param_grids = {
    'logistic_regression': {},
    'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'gradient_boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 10]},
    'svm': {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]},
    'decision_tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'knn': {'n_neighbors': [3, 5, 7]},
    'neural_network': {'hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 20)]},
    'adaboost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'xgboost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 10]}
}

# Define the scorer
scorer = make_scorer(f1_score)

# For each model
for model_name, model in models.items():
    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring=scorer, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Save the model
    joblib.dump(grid_search.best_estimator_, f'models/{model_name}_model.pkl')

    # Evaluate the model
    accuracy = grid_search.best_score_
    print(f'{model_name} Accuracy: {accuracy}')

    # Perform cross-validation
    scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
    print(f'{model_name} Cross-Validation Scores: {scores}') 
