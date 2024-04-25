from sklearn.ensemble import RandomForestClassifier

def create_model(n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators)
    return model
