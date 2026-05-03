import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pickle
import os

# --- Configuration ---
app = Flask(__name__)
MODEL_PATH = 'rf_sales_forecast_model.pkl'
PORT_NUMBER = 5000 # Use 5001 if 5000 is still busy

# --- 1. Load Model and Features ---
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    
    # CRITICAL: Feature list must match the model's training order
    FEATURE_COLUMNS = [
        'Price', 'Discount', 'Competitor Pricing', 'Demand Forecast', 
        'Holiday/Promotion', 'dayofweek', 'dayofyear', 'weekofyear', 
        'month', 'year', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7'
    ]
    print(f"Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    FEATURE_COLUMNS = None

# --- 2. GUI Route (GET /) ---
# This route serves the HTML form to the user.
@app.route('/', methods=['GET'])
def home():
    if model is None:
        return "Error: Sales Forecast Model could not be loaded.", 500
        
    # The HTML template for the input form
    html_template = """
    <!DOCTYPE html>
    <title>Retail Sales Forecast</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input[type="number"], input[type="text"], select { width: 100%; padding: 8px; margin-top: 5px; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 20px; }
        #result { margin-top: 20px; padding: 10px; border: 1px solid #007bff; background-color: #e6f2ff; border-radius: 4px; }
    </style>
    
    <div class="container">
        <h2>Retail Sales Forecast Predictor (Random Forest)</h2>
        <p>Input the daily features below to predict the total units sold for the next day.</p>
        
        <form action="/predict" method="POST">
            <h3>External Features</h3>
            <label>Price ($): <input type="number" step="0.01" name="Price" value="35.0" required></label>
            <label>Discount (%): <input type="number" name="Discount" value="10" required></label>
            <label>Competitor Pricing ($): <input type="number" step="0.01" name="Competitor Pricing" value="34.0" required></label>
            <label>Demand Forecast (Model Output): <input type="number" step="0.01" name="Demand Forecast" value="150.0" required></label>
            <label>Holiday/Promotion (1=Yes, 0=No): <input type="number" name="Holiday/Promotion" value="1" required></label>

            <h3>Date Features</h3>
            <label>Day of Week (0=Monday, 6=Sunday): <input type="number" name="dayofweek" value="1" required></label>
            <label>Day of Year (1-365): <input type="number" name="dayofyear" value="120" required></label>
            <label>Week of Year (1-52): <input type="number" name="weekofyear" value="17" required></label>
            <label>Month (1-12): <input type="number" name="month" value="4" required></label>
            <label>Year: <input type="number" name="year" value="2025" required></label>

            <h3>Historical/Lag Features (Crucial!)</h3>
            <p>These represent the total units sold on previous days.</p>
            <label>Lag 1 (Units Sold Yesterday): <input type="number" name="lag_1" value="15000" required></label>
            <label>Lag 7 (Units Sold 7 Days Ago): <input type="number" name="lag_7" value="14500" required></label>
            <label>Lag 30 (Units Sold 30 Days Ago): <input type="number" name="lag_30" value="16000" required></label>
            <label>Rolling Mean 7 (Avg. Units Sold over last 7 days): <input type="number" step="0.01" name="rolling_mean_7" value="15200" required></label>

            <button type="submit">Get Sales Forecast</button>
        </form>

        {% if result %}
            <div id="result">
                <strong>Forecasted Units Sold:</strong> {{ result }} units
            </div>
        {% endif %}
    </div>
    """
    return render_template_string(html_template)


# --- 3. Prediction Route (POST /predict) ---
# This route handles the form submission and calculation.
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or FEATURE_COLUMNS is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # Get data from the HTML form request
        data = request.form.to_dict()
        
        # Convert all string form values to the correct numeric type (float/int)
        input_data = {
            key: float(value) if key in ['Price', 'Competitor Pricing', 'Demand Forecast', 'rolling_mean_7'] else int(value)
            for key, value in data.items()
        }

        # Create DataFrame and ensure column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        # Generate prediction
        prediction = model.predict(input_df)
        forecast_units = int(np.round(prediction[0]))
        
        # Render the HTML page again, but this time with the result included
        return home(result=forecast_units)

    except KeyError as e:
        return home(error=f'Missing input feature: {e}. Check form fields.')
    except Exception as e:
        return home(error=f'An error occurred during prediction: {e}')


# --- 4. Run the App ---
# NOTE: The home() function is modified to accept a 'result' argument for displaying the forecast.
def home(result=None, error=None):
    # ... (same HTML template structure as above) ...
    # Modified part to show the result or error:
    html_template = """
    ...
    <div class="container">
        ... (form contents) ...
        <button type="submit">Get Sales Forecast</button>
        </form>

        {% if result %}
            <div id="result">
                <strong>Forecasted Units Sold:</strong> {{ result }} units
            </div>
        {% endif %}
        {% if error %}
            <div id="error" style="color: red; margin-top: 20px;">
                <strong>Error:</strong> {{ error }}
            </div>
        {% endif %}
    </div>
    """
    return render_template_string(html_template, result=result, error=error)

if __name__ == '__main__':
    # Use 5001 if 5000 is still unavailable
    print("\n--- FLASK GUI APP STARTED ---")
    print(f"Access the GUI at: http://127.0.0.1:{PORT_NUMBER}/")
    app.run(debug=True, host='0.0.0.0', port=PORT_NUMBER)
