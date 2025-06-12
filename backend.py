from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('wine_mod.pkl', 'rb'))

# Function to return a quality interpretation
def interpret_quality(score):
    if score <= 4:
        return "Bad wine 👎 – Probably not your best pick!"
    elif 5 <= score <= 6:
        return "Average wine 😐 – Might go well with a casual dinner."
    elif 7 <= score <= 8:
        return "Good wine 👍 – Sounds like a tasty choice!"
    else:
        return "Excellent wine 🍷 – Pop that cork and enjoy!"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert all 11 inputs
        input_features = [
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])
        ]

        # Convert to 2D array for prediction
        features = np.array([input_features])
        prediction = model.predict(features)
        predicted_quality = int(round(prediction[0]))
        message = interpret_quality(predicted_quality)

        return render_template('index.html',
                               prediction_text=f'Predicted Wine Quality: {predicted_quality}/10',
                               quality_message=message)

    except Exception as e:
        return render_template('index.html',
                               prediction_text="Something went wrong!",
                               quality_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)

















