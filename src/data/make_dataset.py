import pandas as pd

def add_min_feature():
    # Load the data
    data = pd.read_csv('../../data/raw/customer_data.csv')

    # Create a new feature
    data['X_min'] = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].min(axis=1)

    # Save the data
    data.to_csv('../../data/processed/customer_data.csv', index=False)

add_min_feature()
