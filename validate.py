import pandas as pd
from sklearn.metrics import classification_report
import joblib

def validate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('datasets/test.csv')
    X_test = data[['Ia','Ib','Ic','Va','Vb','Vc']].values
    y_test = data[['G','C','B','A']].values

    # Load and validate your model
    model = joblib.load('model.pkl')
    validate_model(model, X_test, y_test)
