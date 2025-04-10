from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from encodings import feature_mappings

app = Flask(__name__)

# Load your trained model
model = joblib.load('credit_scoring_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        user_data = request.form.to_dict()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([user_data])
        
        # Encode categorical features
        encoded_data = {}
        for feature in feature_mappings:
            if feature in input_df.columns:
                encoded_data[feature] = input_df[feature].map(feature_mappings[feature]).values[0]
            else:
                encoded_data[feature] = 0  # Default value if missing
        
        # Convert to model input format (adjust based on your model)
        features = [
            encoded_data['Education'],
            encoded_data['most_used_financial_service'],
            encoded_data['Area_type'],
            encoded_data['Remittances'],
            encoded_data['Saving'],
            encoded_data['Credit'],
            encoded_data['Region'],
            encoded_data['Income_Sources']
            # Add numerical features if any
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]  # Probability of positive class
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)
