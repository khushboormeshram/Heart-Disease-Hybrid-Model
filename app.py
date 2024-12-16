from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Define the HybridModel class if you used one
class HybridModel:
    def __init__(self, rf_model, lr_model):
        self.rf_model = rf_model
        self.lr_model = lr_model
    
    def predict(self, X):
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        lr_probs = self.lr_model.predict_proba(X)[:, 1]
        hybrid_probs = (rf_probs + lr_probs) / 2
        return (hybrid_probs >= 0.5).astype(int)

# Load the Random Forest and Logistic Regression models
rf_model = joblib.load('models/rf_model.pkl')  # Adjust path as necessary
lr_model = joblib.load('models/lr_model.pkl')  # Adjust path as necessary

# Create the hybrid model
hybrid_model = HybridModel(rf_model, lr_model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create a numpy array from the input data (reshape it to match the model's input shape)
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])

        # Predict using the hybrid model
        prediction = hybrid_model.predict(input_data)

        # Provide output based on the prediction
        if prediction == 1:
            result = "Heart Disease"
        else:
            result = "No Heart Disease"

        # Store the result in a query parameter and redirect back to the home page
        return redirect(url_for('index', prediction_text=f'Prediction: {result}'))

    except Exception as e:
        # Redirect with an error message in case of an exception
        return redirect(url_for('index', prediction_text=f"Error: {str(e)}"))

if __name__ == '__main__':
    app.run(debug=True)
