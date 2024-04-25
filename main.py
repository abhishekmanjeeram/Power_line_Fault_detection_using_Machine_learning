import pandas as pd
import joblib
import numpy as np

def predict(model, X):
    return model.predict(X)

def print_fault_type(predictions):
    fault_types = {
        '1001': 'LG fault (Between Phase A and Gnd)',
        '0011': 'LL fault (Between Phase A and Phase B)',
        '1011': 'LLG Fault (Between Phases A,B and ground)',
        '0111': 'LLL Fault(Between all three phases)',
        '1111': 'LLLG fault( Three phase symmetrical fault)'
    }
    for prediction in predictions:
        prediction_str = ''.join(map(str, prediction.astype(int)))
        print(fault_types.get(prediction_str, 'No Fault'))

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('datasets/input.csv')
    X = data[['Ia','Ib','Ic','Va','Vb','Vc']].values

    # Load your model
    model = joblib.load('model.pkl')

    # Make predictions
    predictions = predict(model, X)

    # Print fault types
    print_fault_type(predictions)
