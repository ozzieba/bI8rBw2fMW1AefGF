import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load the data
data = pd.read_csv('../../data/processed/customer_data.csv')

# Split the data
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
base_models = [
    ('logistic_regression', joblib.load('../../models/logistic_regression_model.pkl')),
    ('random_forest', joblib.load('../../models/random_forest_model.pkl')),
    ('gradient_boosting', joblib.load('../../models/gradient_boosting_model.pkl')),
    ('svm', joblib.load('../../models/svm_model.pkl')),
    ('decision_tree', joblib.load('../../models/decision_tree_model.pkl')),
    ('knn', joblib.load('../../models/knn_model.pkl')),
    ('neural_network', joblib.load('../../models/neural_network_model.pkl')),
    ('adaboost', joblib.load('../../models/adaboost_model.pkl')),
    ('xgboost', joblib.load('../../models/xgboost_model.pkl'))
]

# Define the meta model
meta_model = LogisticRegression()

# Define the stacking classifier
model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, '../../models/stacking_model.pkl')

# Evaluate the base models
for name, base_model in base_models:
    y_pred = base_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}, F1 Score: {f1}')

# Evaluate the stacking model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Stacking Model Accuracy: {accuracy}, F1 Score: {f1}')

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f'Stacking Model Cross-Validation Scores: {scores}')
