import joblib
from sklearn.base import BaseEstimator

class HybridModel(BaseEstimator):
    def __init__(self, rf_model, lr_model):
        self.rf_model = rf_model
        self.lr_model = lr_model

    def fit(self, X, y=None):
        # Fit both models (not strictly needed for hybrid, but you can use this method if needed)
        self.rf_model.fit(X, y)
        self.lr_model.fit(X, y)
        return self

    def predict(self, X):
        # Combine the predictions from both models using the hybrid approach
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        lr_probs = self.lr_model.predict_proba(X)[:, 1]
        hybrid_probs = (rf_probs + lr_probs) / 2
        return [1 if prob >= 0.5 else 0 for prob in hybrid_probs]

    def predict_proba(self, X):
        # Return the averaged probabilities for hybrid model
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        lr_probs = self.lr_model.predict_proba(X)[:, 1]
        hybrid_probs = (rf_probs + lr_probs) / 2
        return hybrid_probs.reshape(-1, 1)

# Assuming rf_model and lr_model are already trained
# Create and fit the hybrid model
hybrid_model = HybridModel(rf_model=rf_model, lr_model=lr_model)
hybrid_model.fit(X_train, y_train)

# Save the hybrid model
joblib.dump(hybrid_model, 'models/hybrid_model.pkl')
