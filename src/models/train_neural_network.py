import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# Load the data
data = pd.read_csv('data/raw/customer_data.csv')

# Split the data
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/neural_network_model.pkl')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')