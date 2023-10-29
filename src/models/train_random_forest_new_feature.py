import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data
data = pd.read_csv('data/raw/customer_data.csv')

# Create a new feature
data['X_avg'] = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis=1)

# Split the data
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/random_forest_with_new_feature_model.pkl')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
