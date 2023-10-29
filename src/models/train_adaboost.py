import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import joblib

# Load the data
data = pd.read_csv('data/raw/customer_data.csv')

# Split the data
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/adaboost_model.pkl')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
