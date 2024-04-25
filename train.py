import pandas as pd
from model import create_model
from sklearn.model_selection import train_test_split, cross_val_score

def train_model(model, X_train, y_train):
    # Perform cross-validation to get an estimate of the model's performance during training
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross-validation scores: {scores}')
    print(f'Average cross-validation score: {scores.mean()}')

    # Train the model on the full training data
    model.fit(X_train, y_train)

if __name__ == "__main__":
    try:
        # Load your data
        data = pd.read_csv('datasets/train.csv')
        X = data[['Ia','Ib','Ic','Va','Vb','Vc']].values
        y = data[['G','C','B','A']].values

        # Split your data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train your model
        model = create_model()
        print("Training model...")
        train_model(model, X_train, y_train)
        print("Model trained successfully!")

        # Save your model
        import joblib
        joblib.dump(model, 'model.pkl')
        print("Model saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
