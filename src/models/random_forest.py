
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf
