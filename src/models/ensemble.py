
import numpy as np
from sklearn.ensemble import VotingClassifier

def train_ensemble(models, X_train, y_train):
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
