import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import your existing call system class and credentials
from voice_call_alert_system import VoiceCallAlertSystem, ACCOUNT_SID, AUTH_TOKEN, TWILIO_PHONE, ALERT_PHONE

# --- NEW: Import modules to run the web server ---
import http.server
import socketserver
import os

# --- Configuration ---
SPHERE_PATIENT_COUNT = 6
TEST_SPLIT_SIZE = 0.3
MODEL_RANDOM_STATE = 42
SERVER_PORT = 8000

def run_ml_and_call():
    """
    This function runs the entire ML pipeline and voice call alert.
    """
    print("Script started. Loading data...")

    # 1. Load the dataset
    try:
        df = pd.read_csv("Health_Risk_Dataset.csv")
        print(f"Dataset '{df.shape[0]}' patients loaded successfully.")
    except FileNotFoundError:
        print("Error: Health_Risk_Dataset.csv not found.")
        print("Please make sure it is in the same folder as this script.")
        return

    # 2. Randomly select 6 "sphere" patients
    # --- MODIFICATION ---
    # Removed 'random_state' to get 6 NEW random patients every time.
    sphere_patients_df = df.sample(n=SPHERE_PATIENT_COUNT)
    print(f"\nRandomly selected {SPHERE_PATIENT_COUNT} new patients for prediction.")

    # 3. Create the training/testing set (everyone *except* the 6 sphere patients)
    train_test_df = df.drop(sphere_patients_df.index)
    print(f"Remaining {len(train_test_df)} patients will be used for training/testing.")

    # 4. Define feature columns
    numerical_features = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 'Systolic_BP', 'Heart_Rate', 'Temperature', 'On_Oxygen']
    categorical_features = ['Consciousness']
    target_column = 'Risk_Level'

    # 5. Preprocess Data: One-hot encode the categorical feature
    print("Preprocessing data (One-Hot Encoding)...")
    train_test_df_processed = pd.get_dummies(train_test_df, columns=categorical_features)
    sphere_patients_df_processed = pd.get_dummies(sphere_patients_df, columns=categorical_features)

    # Get the list of all feature columns (numerical + new one-hot columns)
    all_feature_columns = [col for col in train_test_df_processed.columns if col.startswith('Consciousness_') or col in numerical_features]

    # Align columns: Ensure both dataframes have the exact same one-hot columns
    train_test_df_processed = train_test_df_processed.reindex(columns=all_feature_columns, fill_value=0)
    sphere_patients_df_processed = sphere_patients_df_processed.reindex(columns=all_feature_columns, fill_value=0)

    print("Data processing complete.")

    # 6. Define X (features) and y (target) for the main dataset
    X = train_test_df_processed
    y = train_test_df[target_column] # Target labels are the same

    # 7. Split the main data into 70% training and 30% testing
    print(f"\nSplitting {len(X)} patients into 70/30 train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE,
        random_state=MODEL_RANDOM_STATE, # Use random state here for a consistent model
        stratify=y
    )
    print(f"Training set size: {len(X_train)}")

    # 8. Scale Numerical Features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

    # 9. Train the ML Model
    print("\n🧠 Training the Machine Learning model...")
    model = LogisticRegression(max_iter=1000, random_state=MODEL_RANDOM_STATE)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # 10. Evaluate the Model
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"📊 Model Accuracy (on test set): {test_accuracy * 100:.2f}%")

    # 11. Predict the 6 "Sphere" Patients
    print("\n🔮 Predicting risk for the 6 'sphere' patients...")
    X_sphere_scaled = sphere_patients_df_processed.copy()
    X_sphere_scaled.loc[:, numerical_features] = scaler.transform(X_sphere_scaled[numerical_features])
    sphere_predictions = model.predict(X_sphere_scaled)

    # 12. Store and Report Results
    results_df = sphere_patients_df[['Patient_ID', 'Risk_Level']].copy()
    results_df.rename(columns={'Risk_Level': 'Actual_Risk'}, inplace=True)
    results_df['Predicted_Risk'] = sphere_predictions

    print("\n--- 🔮 Final Predictions for 6 'Sphere' Patients ---")
    print(results_df.to_string(index=False))

    # 13. Save files for other scripts
    results_df.to_csv('patient_predictions.csv', index=False)
    print(f"\n💾 Saved predictions to patient_predictions.csv")
    results_df.to_json('patient_predictions.json', orient='records')
    print(f"💾 Saved predictions to patient_predictions.json")

    # 14. Trigger the Voice Call
    print("\n--- 📞 INITIATING VOICE CALL SYSTEM ---")
    alert_system = VoiceCallAlertSystem(
        account_sid=ACCOUNT_SID,
        auth_token=AUTH_TOKEN,
        twilio_phone=TWILIO_PHONE,
        alert_phone=ALERT_PHONE
    )
    if alert_system.client:
        alert_system.process_patient_predictions(results_df)
    else:
        print("Skipping calls due to Twilio initialization failure.")

def start_web_server():
    """
    This function starts the web server.
    """
    print("\n--- 🖥️  STARTING WEB SERVER ---")
    
    # This makes the server look for files in your project folder
    handler = http.server.SimpleHTTPRequestHandler
    
    # Try to start the server
    try:
        with socketserver.TCPServer(("", SERVER_PORT), handler) as httpd:
            print(f"✅ Server started successfully.")
            print(f"Your dashboard is now running.")
            print("\n" + "="*50)
            print(f"  OPEN THIS URL IN YOUR BROWSER:")
            print(f"  http://localhost:{SERVER_PORT}/dashboard.html")
            print("="*50 + "\n")
            print("(Press Ctrl+C in this terminal to stop the server)")
            httpd.serve_forever()
    except OSError as e:
        print(f"❌ SERVER FAILED TO START (Port {SERVER_PORT} might be in use).")
        print("Please close any other 'http.server' terminals and try again.")
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.socket.close()


# --- Main execution ---
if __name__ == "__main__":
    # Step 1: Run the ML analysis and voice call
    run_ml_and_call()
    
    # Step 2: After the ML part is done, start the web server
    start_web_server()