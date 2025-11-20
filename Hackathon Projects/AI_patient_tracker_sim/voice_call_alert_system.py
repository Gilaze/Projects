"""
VOICE CALL ALERT SYSTEM - Hospital Patient Monitoring
Imports as a class to make calls based on ML predictions.
"""
import time
import pandas as pd
import os
from datetime import datetime
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

# Load environment variables from .env file (override=True to reload fresh values)
load_dotenv(override=True)

# Load Twilio credentials from environment variables
ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE = os.getenv('TWILIO_PHONE_NUMBER')
ALERT_PHONE = os.getenv('TWILIO_ALERT_PHONE')


class VoiceCallAlertSystem:
    def __init__(self, account_sid, auth_token, twilio_phone, alert_phone):
        """Initialize with Twilio credentials"""
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.twilio_phone = twilio_phone
        self.alert_phone = alert_phone
        
        # Initialize Twilio client
        try:
            self.client = Client(account_sid, auth_token)
            self.client.api.v2010.accounts(account_sid).fetch()
            print("Twilio client initialized and credentials verified.")
        except Exception as e:
            print("="*70)
            print("CRITICAL TWILIO ERROR: Could not authenticate.")
            print(f"Error: {e}")
            print("Please check your ACCOUNT_SID and AUTH_TOKEN in 'voice_call_alert_system.py'")
            print("="*70)
            self.client = None

        self.calls_made = []
        
        print("="*70)
        print("VOICE CALL ALERT SYSTEM INITIALIZED")
        print("="*70)
        print(f"Twilio Account: {self.account_sid}")
        print(f"Calling From: {self.twilio_phone}")
        print(f"Calling To: {self.alert_phone}")
        print("="*70)
        print()
    
    # --- THIS IS THE UPDATED FUNCTION ---
    def create_voice_message(self, high_risk_patients):
        """
        Create TwiML voice response with message about high-risk patients
        
        Args:
            high_risk_patients: List of dicts with patient info
        """
        response = VoiceResponse()

        #1 second pause before speaking
        response.pause(length=1)

        # --- Build the patient list string first ---
        spoken_patient_list = []
        for patient in high_risk_patients:
            patient_id = patient['id']
            # Convert P0921 to "Patient P 0 9 2 1"
            spoken_patient_id = f"Patient {', '.join(patient_id)}"
            spoken_patient_list.append(spoken_patient_id)

        # Join the list into a natural-sounding string
        if len(spoken_patient_list) == 0:
            return str(response) # Return empty response
            
        elif len(spoken_patient_list) == 1:
            patient_string = spoken_patient_list[0]
            
        elif len(spoken_patient_list) == 2:
            patient_string = f"{spoken_patient_list[0]} and {spoken_patient_list[1]}"
            
        else:
            # Join all but the last with a comma, then add "and" before the last one
            all_but_last = ", ".join(spoken_patient_list[:-1])
            last = spoken_patient_list[-1]
            patient_string = f"{all_but_last}, and {last}"

        # --- Create the full alert sentence ---
        verb = "is" if len(high_risk_patients) == 1 else "are"
        action = "requires" if len(high_risk_patients) == 1 else "require"
        
        alert_sentence = f"Alert! {patient_string} {verb} at high risk and {action} assistance immediately. "
        
        # === THIS IS THE LOOP YOU WANTED ===
        # It loops the single sentence 3 times
        for i in range(3):
            response.say(
                alert_sentence,
                voice='alice',
                language='en-US'
            )
            # Add a pause between repeats, but not after the last one
            if i < 2:
                response.pause(length=3) # 3-second pause
        # === END OF LOOP ===

        # This is said only ONCE, after the loop is finished
        response.say(
            "End of alert. Goodbye.",
            voice='alice',
            language='en-US'
        )

        return str(response)
    
    def make_voice_call(self, high_risk_patients):
        """
        Make voice call with automated message about high-risk patients
        
        Args:
            high_risk_patients: DataFrame of high-risk patients
        """
        
        if self.client is None:
            print("❌ CALL FAILED: Twilio client not initialized.")
            return

        # Prepare patient list
        patient_list = []
        for idx, patient in high_risk_patients.iterrows():
            patient_list.append({
                'id': patient['Patient_ID'],
                'actual': patient['Actual_Risk'],
                'predicted': patient['Predicted_Risk']
            })
        
        # Create the voice message
        twiml_message = self.create_voice_message(patient_list)
        
        print("📞 PLACING VOICE CALL...")
        print(f"   From: {self.twilio_phone}")
        print(f"   To: {self.alert_phone}")
        print()
        
        # --- MODIFIED: Update the text log to match the new message ---
        print("📢 Voice Message Content (will repeat 3x):")
        print("-" * 70)
        
        spoken_patient_list = [p['id'] for p in patient_list]
        if len(spoken_patient_list) == 1:
            patient_string = spoken_patient_list[0]
            verb = "is"
            action = "requires"
        else:
            patient_string = ", ".join(spoken_patient_list)
            verb = "are"
            action = "require"
            
        print(f"\"Alert! {patient_string} {verb} at high risk and {action} assistance immediately.\"")
        print("-" * 70)
        print()
        
        try:

            # Make the actual call with TwiML
            call = self.client.calls.create(
                twiml=twiml_message,
                to=self.alert_phone,
                from_=self.twilio_phone,
                timeout=60 # Let it ring for 60 seconds
            )
            
            call_record = {
                'timestamp': datetime.now(),
                'call_sid': call.sid,
                'from': self.twilio_phone,
                'to': self.alert_phone,
                'status': call.status,
                'patients': ', '.join([p['id'] for p in patient_list]),
                'num_patients': len(patient_list)
            }
            
            print("✅ CALL INITIATED!")
            print(f"   Call SID: {call.sid}")
            print(f"   Status: {call.status}")
            
            self.calls_made.append(call_record)
            return call_record
            
        except Exception as e:
            print(f"❌ CALL FAILED!")
            print(f"   Error: {e}")
            print("   (Is your ALERT_PHONE a valid, verified number in your Twilio account?)")
            
            call_record = {
                'timestamp': datetime.now(),
                'call_sid': None,
                'status': 'failed',
                'error': str(e),
                'patients': ', '.join([p['id'] for p in patient_list]),
            }
            self.calls_made.append(call_record)
            return call_record
    
    def process_patient_predictions(self, predictions_df):
        """Process predictions and make voice call for high-risk patients"""
        print("="*70)
        print("PROCESSING PATIENT RISK PREDICTIONS FOR CALLS")
        print("="*70)
        
        high_risk_patients = predictions_df[predictions_df['Predicted_Risk'] == 'High']
        
        if len(high_risk_patients) == 0:
            print("✅ No high-risk patients detected. No calls needed.")
            return
        
        print(f"⚠️  ALERT: {len(high_risk_patients)} HIGH-RISK PATIENT(S) DETECTED!")
        print()
        print("High-Risk Patients:")
        for idx, patient in high_risk_patients.iterrows():
            print(f"   • {patient['Patient_ID']} (Actual: {patient['Actual_Risk']})")
        print()
        

        # Make the voice call
        self.make_voice_call(high_risk_patients)
        
        print("="*70)
        print(f"SUMMARY: Voice call {'placed' if self.calls_made else 'attempted'}")
        print("="*70)

# This allows the script to be run by itself for testing
if __name__ == "__main__":
    print("Running voice_call_alert_system.py in standalone test mode...")
    
    # Create dummy prediction data
    dummy_data = {
        'Patient_ID': ['P-TEST-1', 'P-TEST-2', 'P-TEST-3'],
        'Actual_Risk': ['High', 'Low', 'High'],
        'Predicted_Risk': ['High', 'Low', 'High']
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    # Initialize system
    alert_system = VoiceCallAlertSystem(
        account_sid=ACCOUNT_SID,
        auth_token=AUTH_TOKEN,
        twilio_phone=TWILIO_PHONE,
        alert_phone=ALERT_PHONE
    )
    
    # Process and call
    if alert_system.client:
        alert_system.process_patient_predictions(dummy_df)
    else:
        print("Exiting test mode due to Twilio initialization failure.")