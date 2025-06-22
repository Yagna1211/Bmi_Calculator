# === app.py ===
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('BMI_CALCI.pkl')

labels = {
    0: ("Extremely Weak", "😓", "You need urgent care!"),
    1: ("Weak", "🥺", "You should gain some healthy weight."),
    2: ("Normal", "🎉", "Great! You're fit and fabulous!"),
    3: ("Overweight", "😬", "Consider watching your diet."),
    4: ("Obesity", "⚠️", "High health risk! Seek guidance."),
    5: ("Extreme Obesity", "🚨", "Critical! Immediate action required.")
}

recommendations = {
    "Extremely Weak": {
        "diet": "Eat calorie-rich foods like nuts, dairy, and whole grains.",
        "exercise": "Focus on light strength training and yoga."
    },
    "Weak": {
        "diet": "Add protein and complex carbs. Eat more frequent meals.",
        "exercise": "Go for brisk walking and light muscle exercises."
    },
    "Normal": {
        "diet": "Maintain a balanced diet with fruits, veggies, and lean proteins.",
        "exercise": "Keep up regular cardio and resistance training."
    },
    "Overweight": {
        "diet": "Cut down on sugar and oily food. Add more fiber.",
        "exercise": "Try HIIT, cardio, and 30 mins walking daily."
    },
    "Obesity": {
        "diet": "Follow a low-carb diet. Avoid fried and fast food.",
        "exercise": "Start with walking. Gradually include cycling or swimming."
    },
    "Extreme Obesity": {
        "diet": "Seek help from a registered dietitian. Avoid all junk foods.",
        "exercise": "Start supervised physiotherapy. No intense workouts initially."
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])

    gender_encoded = 1 if gender.lower() == 'male' else 0
    features = np.array([[gender_encoded, height, weight]])

    index = int(model.predict(features)[0])
    category, emoji, message = labels.get(index, ("Unknown", "❓", "No data."))
    rec = recommendations.get(category, {})

    return render_template('index.html',
                           category=category,
                           emoji=emoji,
                           message=message,
                           diet=rec.get("diet", ""),
                           exercise=rec.get("exercise", ""))

if __name__ == '__main__':
    app.run(debug=True)
