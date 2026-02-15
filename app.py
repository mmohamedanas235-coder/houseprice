from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['longitude']),
            float(request.form['latitude']),
            float(request.form['housing_median_age']),
            float(request.form['total_rooms']),
            float(request.form['total_bedrooms']),
            float(request.form['population']),
            float(request.form['households']),
            float(request.form['median_income']),
            encoder.transform([request.form['ocean_proximity']])[0]
        ]

        prediction = model.predict([data])[0]
        return render_template('index.html', prediction_text=f'Predicted Median House Value: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
