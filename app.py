"""
PropIQ Backend - House Price Prediction API
Loads a pre-trained ML model from pickle and provides predictions
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# SKLEARN COMPATIBILITY PATCH (REQUIRED)
# ============================================================
# Patch for sklearn 1.8.0 loading pickles from 1.6.1
try:
    from sklearn.compose._column_transformer import _RemainderColsList
except (ImportError, AttributeError):
    # If _RemainderColsList doesn't exist, create a stub
    class _RemainderColsList:
        pass
    import sklearn.compose._column_transformer as ct
    ct._RemainderColsList = _RemainderColsList

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# ============================================================
# MODEL LOADING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    'Price_predict.pickle',
    'Price_Predict.pickle',
    'finmodel.pkl',
    'finalmmm.pkl',
    'finalmm.pkl',
    'finalm.pkl',
    'final.pkl',
]

MODEL_PATH = None
for candidate in MODEL_CANDIDATES:
    candidate_path = os.path.join(BASE_DIR, candidate)
    if os.path.exists(candidate_path):
        MODEL_PATH = candidate_path
        break

ACTIVE_MODEL_NAME = os.path.basename(MODEL_PATH) if MODEL_PATH else None

print(f"\n{'='*60}")
print("PropIQ - Loading Model")
print(f"{'='*60}")
print(f"Model candidates: {MODEL_CANDIDATES}")
print(f"Selected model path: {MODEL_PATH}")
print(f"Active model name: {ACTIVE_MODEL_NAME}")
print(f"Model exists: {bool(MODEL_PATH and os.path.exists(MODEL_PATH))}")

model = None
try:
    if MODEL_PATH is None:
        raise FileNotFoundError(
            f"No model file found. Tried: {', '.join(MODEL_CANDIDATES)}"
        )
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded successfully from: {MODEL_PATH}")
    print(f"[OK] Model type: {type(model)}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()



# ============================================================
# FEATURE ENCODING
# ============================================================

# Location one-hot columns expected by Price_predict.pickle.
LOCATION_COLUMNS = [
    'Banjara Hills', 'Gachibowli', 'HITEC City', 'Jubilee Hills', 'Kokapet',
    'Kompally', 'Kondapur', 'Kukatpally', 'LB Nagar', 'Madhapur',
    'Manikonda', 'Medchal', 'Mehdipatnam', 'Miyapur', 'Moinabad',
    'Nallagandla', 'Narsingi', 'Secunderabad', 'Uppal'
]

PROPERTY_TYPE_MAP = {'Apartment': 0, 'Independent House': 1, 'Villa': 2}
FURNISHING_STATUS_MAP = {'Furnished': 0, 'Semi-Furnished': 1, 'Unfurnished': 2}
LIFT_AVAILABILITY_MAP = {'No': 0, 'Yes': 1}
DIRECTION_FACING_MAP = {
    'East': 0, 'North': 1, 'South': 2, 'West': 3,
    # Graceful fallback for frontend diagonal options.
    'North-East': 1, 'North-West': 1, 'South-East': 2, 'South-West': 2
}

def uses_location_encoded_features():
    if model is None or not hasattr(model, 'feature_names_in_'):
        return False
    return any(col in model.feature_names_in_ for col in LOCATION_COLUMNS)

def prediction_to_inr(raw_prediction):
    """
    Convert model output to INR based on the active model's target scale.
    Price_predict*.pickle models already return full INR values.
    Some fin*/final* models return values in crores and need scaling.
    """
    crore_scaled_models = {'finmodel.pkl', 'finalm.pkl', 'finalmm.pkl', 'finalmmm.pkl', 'final.pkl'}
    if ACTIVE_MODEL_NAME in crore_scaled_models:
        return raw_prediction * 10000000.0
    return raw_prediction

def build_model_input(data):
    """
    Transform frontend payload to the exact feature shape expected by the loaded model.
    """
    try:
        print(f"\n[DEBUG] Building model input...")
        
        # Clip numeric features to safe training bounds to reduce OOD extrapolation.
        area_sqft = max(400.0, min(6000.0, float(data.get('Area_sqft', 0))))
        bedrooms = int(max(1, min(6, int(data.get('Bedrooms', 0)))))
        bathrooms = int(max(1, min(6, int(data.get('Bathrooms', 0)))))
        year_built = int(data.get('Year_Built', 2000))
        parking_spaces = int(data.get('Parking_Spaces', 0))
        floor_number = int(data.get('Floor_Number', 0))
        distance_to_city = max(0.5, min(50.0, float(data.get('Distance_to_City_Center_km', 0))))
        balcony_count = int(data.get('Balcony_Count', 0))
        
        # Extract 5 categorical features
        location = data.get('Location', '')
        property_type = data.get('Property_Type', '')
        furnishing_status = data.get('Furnishing_Status', '')
        lift_availability = data.get('Lift_Availability', '')
        direction_facing = data.get('Direction_Facing', '')

        property_age = 2026 - year_built
        beds_baths = bedrooms + bathrooms
        location_demand = len(location)
        area_location = area_sqft * location_demand

        if uses_location_encoded_features():
            expected_features = list(model.feature_names_in_)
            row = {feature: 0 for feature in expected_features}

            row['Area_sqft'] = area_sqft
            row['Bedrooms'] = bedrooms
            row['Bathrooms'] = bathrooms
            row['Year_Built'] = year_built
            row['Parking_Spaces'] = parking_spaces
            row['Floor_Number'] = floor_number
            row['Distance_to_City_Center_km'] = distance_to_city
            row['Balcony_Count'] = balcony_count

            row['Property_Type'] = PROPERTY_TYPE_MAP.get(property_type, 0)
            row['Furnishing_Status'] = FURNISHING_STATUS_MAP.get(furnishing_status, 0)
            row['Lift_Availability'] = LIFT_AVAILABILITY_MAP.get(lift_availability, 1)
            row['Direction_Facing'] = DIRECTION_FACING_MAP.get(direction_facing, 0)

            if location in row:
                row[location] = 1

            df = pd.DataFrame([row], columns=expected_features)
        else:
            # Supports pipeline models such as finalm.pkl that expect raw
            # categoricals plus engineered features.
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
                expected_features = list(model.feature_names_in_)
                df = pd.DataFrame([{feature: base_row.get(feature, 0) for feature in expected_features}],
                                  columns=expected_features)
            else:
                df = pd.DataFrame([base_row])
        
        print(f"[DEBUG] Raw feature DataFrame created:")
        print(f"        Total features: {len(df.columns)}")
        print(f"        Shape: {df.shape}")
        
        return df
    
    except Exception as e:
        print(f"\n[ERROR] Feature building error: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_input(data):
    """
    Preprocess input data from the frontend.
    Validates required fields and calls build_model_input for feature transformation.
    """
    try:
        print(f"\n[DEBUG] Input JSON received:")
        print(json.dumps(data, indent=2))
        
        # Validate required fields
        required_fields = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Year_Built', 
                          'Parking_Spaces', 'Floor_Number', 'Distance_to_City_Center_km',
                          'Balcony_Count', 'Location', 'Property_Type', 
                          'Furnishing_Status', 'Lift_Availability', 'Direction_Facing']
        
        missing = [field for field in required_fields if field not in data or data[field] == '']
        if missing:
            return None, f"Missing required fields: {', '.join(missing)}"
        
        # Build model-ready feature input
        features_df = build_model_input(data)
        
        return features_df, None
    
    except Exception as e:
        print(f"\n[ERROR] Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Preprocessing error: {str(e)}"

# ============================================================
# ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """Health check / info endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'PropIQ House Price Prediction API',
        'model_loaded': model is not None,
        'active_model': ACTIVE_MODEL_NAME,
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Expects JSON payload with 13 house features.
    Returns predicted price in INR.
    """
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Check server logs.',
            'success': False
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            }), 400

        # Preprocess input - validates and transforms to 45 features
        features_df, error = preprocess_input(data)
        
        if error:
            return jsonify({
                'error': error,
                'success': False
            }), 400
        
        print(f"\n[DEBUG] Calling model.predict() with {len(features_df.columns)} features...")
        try:
            prediction = model.predict(features_df)[0]
            print(f"[DEBUG] Raw prediction output: {prediction}")
            print(f"[DEBUG] Prediction type: {type(prediction)}")
        except Exception as e:
            print(f"\n[ERROR] Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Validate prediction
        if np.isnan(prediction) or np.isinf(prediction):
            return jsonify({
                'error': 'Model returned invalid prediction (NaN or Inf)',
                'success': False
            }), 500
        
        # 1. Prevent negative prices
        raw_prediction = max(0, float(prediction))

        # 2. Convert model-specific output scale to INR and remove decimals.
        predicted_price = int(prediction_to_inr(raw_prediction))

        # 3. Sanity guard: cap unrealistic predictions to 100 crores.
        max_price_ceiling = 1000000000
        # Keep model variation; only apply hard cap to outliers.
        predicted_price = min(predicted_price, max_price_ceiling)
        
        print(f"\n[SUCCESS] Predicted price: INR {predicted_price:,.2f}")
        
        return jsonify({
            'predicted_price': predicted_price,
            'price': predicted_price,
            'currency': 'INR',
            'success': True
        }), 200
    
    except Exception as e:
        print(f"\n[ERROR] Prediction endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'active_model': ACTIVE_MODEL_NAME
    }), 200 if model is not None else 503

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'success': False}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed', 'success': False}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'success': False}), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("PropIQ - House Price Prediction Backend")
    print(f"{'='*60}")
    print(f"Model Status: {'LOADED' if model is not None else 'NOT LOADED'}")
    print(f"API Endpoint: http://127.0.0.1:5000/predict")
    print(f"Input Features: 13 raw (8 numeric + 5 categorical)")
    if model is not None and hasattr(model, 'feature_names_in_'):
        print(f"Model Features: {len(model.feature_names_in_)} expected columns")
    else:
        print("Model Features: unknown (no feature_names_in_)")
    print(f"{'='*60}\n")
    
    # Run Flask app in debug mode
    app.run(debug=True, host='127.0.0.1', port=5000)