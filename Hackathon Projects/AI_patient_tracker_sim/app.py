import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# Import your call system
from voice_call_alert_system import VoiceCallAlertSystem, ACCOUNT_SID, AUTH_TOKEN, TWILIO_PHONE, ALERT_PHONE

# Import Flask
from flask import Flask, jsonify, send_from_directory

# --- Configuration ---
SPHERE_PATIENT_COUNT = 6
TEST_SPLIT_SIZE = 0.3
MODEL_RANDOM_STATE = 42
SERVER_PORT = 8000
DATASET_FILE = "Health_Risk_Dataset.csv"

# --- 1. Create the Flask App ---
app = Flask(__name__)
print("Flask app initializing...")

# --- 2. Create the ML/Call Function ---
# This is all your logic, now inside a function
def run_new_analysis():
    """
    Loads data, picks 6 random patients, trains, predicts,
    and makes a call.
    Returns the 6 patient predictions as a JSON-friendly list.
    """
    print("\n[SERVER LOG] === RUNNING NEW ANALYSIS ===")
    
    # 1. Load data
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"[SERVER LOG] ERROR: {DATASET_FILE} not found.")
        return [] # Return empty list on error

    # 2. Get 6 new random patients
    sphere_patients_df = df.sample(n=SPHERE_PATIENT_COUNT)
    print(f"[SERVER LOG] Selected 6 new random patients.")

    # 3. Create training set (everyone else)
    train_test_df = df.drop(sphere_patients_df.index)

    # 4. Define features
    numerical_features = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 'Systolic_BP', 'Heart_Rate', 'Temperature', 'On_Oxygen']
    categorical_features = ['Consciousness']
    target_column = 'Risk_Level'

    # 5. Preprocess Data
    print("[SERVER LOG] Preprocessing data...")
    train_test_df_processed = pd.get_dummies(train_test_df, columns=categorical_features)
    sphere_patients_df_processed = pd.get_dummies(sphere_patients_df, columns=categorical_features)
    all_feature_columns = [col for col in train_test_df_processed.columns if col.startswith('Consciousness_') or col in numerical_features]
    train_test_df_processed = train_test_df_processed.reindex(columns=all_feature_columns, fill_value=0)
    sphere_patients_df_processed = sphere_patients_df_processed.reindex(columns=all_feature_columns, fill_value=0)

    # 6. Define X and y
    X = train_test_df_processed
    y = train_test_df[target_column]

    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=MODEL_RANDOM_STATE, stratify=y
    )

    # 8. Scale
    print("[SERVER LOG] Training model...")
    scaler = StandardScaler()
    X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

    # 9. Train
    model = LogisticRegression(max_iter=1000, random_state=MODEL_RANDOM_STATE)
    model.fit(X_train, y_train)

    # 10. Predict the 6 spheres
    print("[SERVER LOG] Predicting 6 patients...")
    X_sphere_scaled = sphere_patients_df_processed.copy()
    X_sphere_scaled.loc[:, numerical_features] = scaler.transform(X_sphere_scaled[numerical_features])
    sphere_predictions = model.predict(X_sphere_scaled)

    # 11. Store results
    results_df = sphere_patients_df[['Patient_ID', 'Risk_Level']].copy()
    results_df.rename(columns={'Risk_Level': 'Actual_Risk'}, inplace=True)
    results_df['Predicted_Risk'] = sphere_predictions
    
    print("[SERVER LOG] Predictions complete:")
    print(results_df.to_string(index=False))

    # 12. Trigger Voice Call
    print("[SERVER LOG] Initializing call system...")
    alert_system = VoiceCallAlertSystem(
        account_sid=ACCOUNT_SID,
        auth_token=AUTH_TOKEN,
        twilio_phone=TWILIO_PHONE,
        alert_phone=ALERT_PHONE
    )
    if alert_system.client:
        alert_system.process_patient_predictions(results_df)
    
    # 13. Return predictions as JSON
    # This is what we send back to the browser
    return results_df.to_dict('records')


# --- 3. Create the API Endpoints (The "Routes") ---

@app.route('/')
def serve_dashboard():
    """
    This is the main page. It serves your dashboard.html file.
    """
    print("[SERVER LOG] Dashboard.html requested by user.")
    return send_from_directory('.', 'dashboard.html')

@app.route('/run-analysis', methods=['POST'])
def handle_run_analysis():
    """
    This is the API endpoint your button will call.
    It runs the ML function and returns the new patient data.
    """
    # Run the big function
    new_patient_data = run_new_analysis()
    
    # Send the new data back to the browser as JSON
    return jsonify(new_patient_data)

@app.route('/<path:filename>')
def serve_static(filename):
    """
    This route serves any other files the dashboard might need,
    like CSS or your dataset CSV (if the dashboard used it).
    """
    return send_from_directory('.', filename)

# --- 4. Run the Server ---
if __name__ == "__main__":
    print("="*50)
    print(f"  Starting Server...")
    print(f"  OPEN THIS URL IN YOUR BROWSER:")
    print(f"  http://localhost:{SERVER_PORT}")
    print("="*50 + "\n")
    print("(Press Ctrl+C in this terminal to stop the server)")
    app.run(port=SERVER_PORT, debug=False)