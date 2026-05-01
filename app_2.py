"""
PropIQ Backend - House Price Prediction API (Production Version)
Includes price rounding, sanity checks, and input clipping.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Sklearn compatibility patch
try:
    from sklearn.compose._column_transformer import _RemainderColsList
except (ImportError, AttributeError):
    class _RemainderColsList:
        pass
    import sklearn.compose._column_transformer as ct
    ct._RemainderColsList = _RemainderColsList

app = Flask(__name__)
CORS(app)

# Model Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'finalmm.pkl')

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded successfully: {MODEL_PATH}")
    else:
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Loading error: {e}")

# Mappings for categorical features
PROPERTY_TYPE_MAP = {'Apartment': 0, 'Independent House': 0.5, 'Villa': 1.0}

def build_model_input(data):
    """
    Builds model input with hard clipping to prevent trillion-dollar extrapolation errors.
    """
    # Input Clipping / Sanity Guard for raw features
    area_sqft = np.clip(float(data.get('Area_sqft', 0)), 400, 6000)
    bedrooms = np.clip(int(data.get('Bedrooms', 0)), 1, 6)
    bathrooms = np.clip(int(data.get('Bathrooms', 0)), 1, 6)
    year_built = int(data.get('Year_Built', 2020))
    parking_spaces = int(data.get('Parking_Spaces', 0))
    floor_number = int(data.get('Floor_Number', 0))
    distance_to_city = np.clip(float(data.get('Distance_to_City_Center_km', 0)), 0.5, 50)
    balcony_count = int(data.get('Balcony_Count', 0))
    
    location = data.get('Location', '')
    property_type = data.get('Property_Type', '')
    furnishing_status = data.get('Furnishing_Status', '')
    lift_availability = data.get('Lift_Availability', '')
    direction_facing = data.get('Direction_Facing', '')

    # Derived Features
    property_age = 2026 - year_built
    beds_baths = bedrooms + bathrooms
    location_demand = len(location)
    area_location = area_sqft * location_demand

    base_row = {
        'Area_sqft': area_sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Year_Built': year_built,
        'Parking_Spaces': parking_spaces,
        'Floor_Number': floor_number,
        'Distance_to_City_Center_km': distance_to_city,
        'Balcony_Count': balcony_count,
        'Location': location,
        'Property_Type': property_type,
        'Furnishing_Status': furnishing_status,
        'Lift_Availability': lift_availability,
        'Direction_Facing': direction_facing,
        'Property_Age': property_age,
        'Beds_Baths': beds_baths,
        'Location_Demand': location_demand,
        'Area_Location': area_location,
    }

    if model is not None and hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        df = pd.DataFrame([{f: base_row.get(f, 0) for f in expected}], columns=expected)
    else:
        df = pd.DataFrame([base_row])
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model unavailable', 'success': False}), 500
    
    try:
        data = request.get_json()
        features_df = build_model_input(data)
        
        # Predict (Raw output is in Crores)
        prediction = model.predict(features_df)[0]
        
        # Sanity Guard: Prevent negative or astronomical prices
        # Cap at 100 Crores
        prediction = np.clip(prediction, 0, 100.0)
        
        # Convert to INR and Round to nearest whole number
        predicted_price_inr = int(round(prediction * 10000000))
        
        return jsonify({
            'predicted_price': predicted_price_inr,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
"""
