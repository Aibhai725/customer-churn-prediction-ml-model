from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model_1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load LabelEncoder dictionary
with open('encoders_1.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Columns in the same order as model training
feature_columns = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

# Separate numeric and categorical columns
numeric_cols = ['Time_spent_Alone', 'Going_outside', 'Social_event_attendance', 
                'Friends_circle_size', 'Post_frequency']
categorical_cols = [col for col in feature_columns if col not in numeric_cols]

# Function to encode categorical columns
def encode_feature(col_name, value):
    le = encoders[col_name]
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return le.transform([le.classes_[0]])[0]  # most frequent class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for col in feature_columns:
            val = request.form[col]
            if col in numeric_cols:
                features.append(int(val))
            else:
                features.append(encode_feature(col, val))

        input_df = pd.DataFrame([features], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        result = "🌙 The person is Introvert " if prediction == 1 else "😎The person is Extrovert"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
