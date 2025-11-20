# 🚨 GOT A 401 ERROR? - QUICK FIX

## The Error You Saw:
```
[ERROR] API Error: 401 - authentication_error
x-api-key header is required
```

## The Fix (2 Minutes):

### Step 1: Get an API Key
Go to: **https://console.anthropic.com/**
- Sign up (it's free to start!)
- Create an API key
- Copy it (starts with `sk-ant-api03-...`)

### Step 2: Set the API Key

**Windows:**
```cmd
setx ANTHROPIC_API_KEY "your-key-here"
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Step 3: Restart Terminal & Run Again
```bash
python media_detector_enhanced.py
```

---

## ✅ That's It!

**Need more help?** Read: [API_KEY_SETUP.md](API_KEY_SETUP.md)

**Quick Summary:**
- The detector uses Claude's AI to analyze screenshots
- Claude API requires authentication (the API key)
- New accounts get **free credits** to try it out!
- After setup, it works perfectly

---

## 💰 About Costs

- **Free credits:** New users get $5-10 free
- **Per check:** Less than $0.003 (0.3 cents)
- **Hourly usage:** About $0.36-1.08/hour
- **You control it:** Adjust check frequency in config.py

Don't want to pay? Check out the free alternatives section in API_KEY_SETUP.md!

---

**Go get your API key now and start protecting yourself from fake content!** 🛡️
