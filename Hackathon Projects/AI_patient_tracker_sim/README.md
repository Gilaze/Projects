Using Claude Code, Sonnet 4.5, and Twilio API (for real life calling), my team and I were able to create a hospital simulation where an AI system monitors patient vitals and uses XGBoost to predict if a patient is at high risk. 

For every patient at high risk detected, the program generates a REAL LIFE PHONE CALL (tested with my personal phone number) with a BOT MESSAGE (i.e. "Patient x,y,z are at high risk and require assistance immediately!"). The bot message specifically lists the SIMULATED patients with high risk by using concatenation and Twilio's 'say' function. 
