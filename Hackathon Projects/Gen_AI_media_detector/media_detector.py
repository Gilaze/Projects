#!/usr/bin/env python3
"""
Media Authenticity Detector
Monitors screen content and alerts users to AI-generated or misleading media
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import base64
import json
from io import BytesIO
from PIL import ImageGrab, Image
import requests
from datetime import datetime

import os

# Get API key from environment variable
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

if not ANTHROPIC_API_KEY:
    print("=" * 60)
    print("ERROR: No API key found!")
    print("=" * 60)
    print()
    print("You need a Claude API key to use this tool.")
    print()
    print("Get your API key from: https://console.anthropic.com/")
    print()
    print("Set it as an environment variable:")
    print('  Windows:     setx ANTHROPIC_API_KEY "your-key-here"')
    print('  Mac/Linux:   export ANTHROPIC_API_KEY="your-key-here"')
    print()
    print("After setting, restart this program.")
    print("=" * 60)
    exit(1)


import re

class MediaDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main window
        self.monitoring = False
        self.last_check_time = 0
        self.check_interval = 3  # Check every 3 seconds
        self.notification_windows = []
        
    def capture_screen(self):
        """Capture the current screen"""
        try:
            screenshot = ImageGrab.grab()
            return screenshot
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        # Resize image to reduce token usage
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def analyze_with_claude(self, image_base64):
        """Analyze image using Claude API to detect AI content and misinformation"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": """Analyze this screenshot for potential AI-generated content or misinformation. 

CRITICAL: Respond ONLY with valid JSON. Do not include any text outside the JSON structure, including backticks or explanations.

Your response must be a single JSON object with this exact structure:
{
  "has_media": true/false,
  "is_ai_generated": true/false,
  "ai_confidence": 0-100,
  "has_text_content": true/false,
  "extracted_text": "text here or empty string",
  "is_news": true/false,
  "is_misleading": true/false,
  "misleading_confidence": 0-100,
  "alert_type": "none/ai_generated/misleading/both",
  "reason": "brief explanation"
}

Detection criteria:
- Look for social media posts (Instagram, TikTok, Facebook, YouTube, Twitter/X, etc.)
- Check for AI artifacts: unnatural faces, weird hands, inconsistent lighting, text errors
- Check if there's news content or claims that can be fact-checked
- Identify political content, propaganda, or sensational claims
- DO NOT flag professional graphics, UI elements, or legitimate content as AI

Be conservative - only flag content you're reasonably confident about."""
                                }
                            ]
                        }
                    ]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                result_text = data['content'][0]['text'].strip()
                # Remove markdown code blocks if present
                result_text = re.sub(r'```json\s*', '', result_text)
                result_text = re.sub(r'```\s*', '', result_text)
                result_text = result_text.strip()
                
                result = json.loads(result_text)
                return result
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {result_text}")
            return None
        except Exception as e:
            print(f"Error analyzing with Claude: {e}")
            return None
    
    def fact_check_claim(self, claim_text):
        """Fact-check a claim by searching and comparing with news sources"""
        try:
            # First, search for the claim
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1500,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""I need to fact-check this claim: "{claim_text}"

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

Respond with only a JSON object in this format:
{{
  "main_claims": ["claim1", "claim2"],
  "search_queries": ["query1", "query2"]
}}

Extract the key factual claims and generate 2-3 search queries to verify them."""
                        }
                    ]
                }
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            result_text = data['content'][0]['text'].strip()
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            
            query_data = json.loads(result_text)
            search_queries = query_data.get('search_queries', [])
            
            if not search_queries:
                return None
            
            # Use first search query to find sources
            search_query = search_queries[0]
            
            # Now analyze with sources (simplified - in production you'd use web_search)
            fact_check_response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Based on your knowledge, fact-check this claim: "{claim_text}"

RESPOND ONLY WITH VALID JSON. NO OTHER TEXT.

{{
  "verdict": "true/false/misleading/unverifiable",
  "confidence": 0-100,
  "explanation": "brief explanation",
  "context_needed": true/false
}}"""
                        }
                    ]
                }
            )
            
            if fact_check_response.status_code != 200:
                return None
            
            fact_data = fact_check_response.json()
            fact_text = fact_data['content'][0]['text'].strip()
            fact_text = re.sub(r'```json\s*', '', fact_text)
            fact_text = re.sub(r'```\s*', '', fact_text)
            
            return json.loads(fact_text)
            
        except Exception as e:
            print(f"Error fact-checking: {e}")
            return None
    
    def get_context(self, claim_text):
        """Get detailed context and debunking information"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Provide context to debunk or clarify this misleading claim: "{claim_text}"

Include:
1. What the actual facts are
2. Why this claim is misleading
3. Key sources or evidence that disprove it

Keep it concise (3-5 bullet points) and accessible."""
                        }
                    ]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text']
            return None
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return None
    
    def show_notification(self, alert_type, message, claim_text=None):
        """Show a notification window that auto-dismisses after 5 seconds"""
        notification = tk.Toplevel()
        notification.withdraw()  # Hide initially
        
        # Configure window
        notification.overrideredirect(True)  # Remove window decorations
        notification.attributes('-topmost', True)  # Always on top
        
        # Set color based on alert type
        if alert_type == "ai_generated":
            bg_color = "#FF6B6B"  # Red
            title_text = "⚠️ AI-Generated Content Detected"
        elif alert_type == "misleading":
            bg_color = "#FFA500"  # Orange
            title_text = "⚠️ Potentially Misleading Information"
        else:  # both
            bg_color = "#DC143C"  # Crimson
            title_text = "⚠️ AI-Generated & Misleading Content"
        
        notification.configure(bg=bg_color)
        
        # Create frame
        frame = tk.Frame(notification, bg=bg_color, padx=15, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            frame, 
            text=title_text,
            font=('Arial', 12, 'bold'),
            bg=bg_color,
            fg='white'
        )
        title_label.pack(anchor='w')
        
        # Message
        msg_label = tk.Label(
            frame,
            text=message,
            font=('Arial', 10),
            bg=bg_color,
            fg='white',
            wraplength=350,
            justify='left'
        )
        msg_label.pack(anchor='w', pady=(5, 0))
        
        # Context button for misleading content
        if claim_text and alert_type in ["misleading", "both"]:
            def show_context():
                # Disable button
                context_btn.config(state='disabled', text='Loading...')
                notification.update()
                
                # Get context in background
                context = self.get_context(claim_text)
                
                if context:
                    # Create context window
                    context_window = tk.Toplevel()
                    context_window.title("Context & Fact-Check")
                    context_window.geometry("500x400")
                    context_window.attributes('-topmost', True)
                    
                    # Text widget with scrollbar
                    text_frame = tk.Frame(context_window)
                    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    scrollbar = tk.Scrollbar(text_frame)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    text_widget = tk.Text(
                        text_frame,
                        wrap=tk.WORD,
                        yscrollcommand=scrollbar.set,
                        font=('Arial', 10),
                        padx=10,
                        pady=10
                    )
                    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    scrollbar.config(command=text_widget.yview)
                    
                    text_widget.insert('1.0', context)
                    text_widget.config(state='disabled')
                    
                    close_btn = tk.Button(
                        context_window,
                        text="Close",
                        command=context_window.destroy,
                        font=('Arial', 10)
                    )
                    close_btn.pack(pady=5)
                else:
                    context_btn.config(state='normal', text='Context')
            
            context_btn = tk.Button(
                frame,
                text="Context",
                command=show_context,
                bg='white',
                fg=bg_color,
                font=('Arial', 9, 'bold'),
                relief=tk.RAISED,
                cursor='hand2'
            )
            context_btn.pack(anchor='w', pady=(8, 0))
        
        # Position at top center of screen
        notification.update_idletasks()
        width = notification.winfo_reqwidth()
        screen_width = notification.winfo_screenwidth()
        x = (screen_width - width) // 2
        y = 20
        
        notification.geometry(f"+{x}+{y}")
        notification.deiconify()  # Show window
        
        # Auto-dismiss after 5 seconds
        def dismiss():
            try:
                notification.destroy()
                if notification in self.notification_windows:
                    self.notification_windows.remove(notification)
            except:
                pass
        
        notification.after(5000, dismiss)
        self.notification_windows.append(notification)
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            current_time = time.time()
            
            if current_time - self.last_check_time >= self.check_interval:
                self.last_check_time = current_time
                
                # Capture and analyze screen
                screenshot = self.capture_screen()
                if screenshot:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing screen...")
                    
                    image_base64 = self.image_to_base64(screenshot)
                    analysis = self.analyze_with_claude(image_base64)
                    
                    if analysis:
                        print(f"Analysis: {json.dumps(analysis, indent=2)}")
                        
                        alert_type = analysis.get('alert_type', 'none')
                        
                        if alert_type != 'none':
                            message = analysis.get('reason', 'Suspicious content detected')
                            claim_text = analysis.get('extracted_text', '')
                            
                            # If it's news/misleading, do additional fact-checking
                            if analysis.get('is_misleading') and claim_text:
                                fact_check = self.fact_check_claim(claim_text)
                                if fact_check:
                                    verdict = fact_check.get('verdict', 'unverifiable')
                                    if verdict in ['false', 'misleading']:
                                        message += f"\n\nFact-check: {fact_check.get('explanation', '')}"
                            
                            # Show notification on main thread
                            self.root.after(0, lambda: self.show_notification(
                                alert_type, 
                                message,
                                claim_text if analysis.get('is_misleading') else None
                            ))
            
            time.sleep(0.5)  # Small sleep to prevent CPU spinning
    
    def start_monitoring(self):
        """Start monitoring the screen"""
        if not self.monitoring:
            self.monitoring = True
            print("🔍 Media Detector started! Monitoring your screen...")
            print("Press Ctrl+C to stop.\n")
            
            # Start monitoring in background thread
            monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            monitor_thread.start()
            
            # Run tkinter main loop
            self.root.mainloop()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("\n✅ Media Detector stopped.")

def main():
    detector = MediaDetector()
    
    try:
        detector.start_monitoring()
    except KeyboardInterrupt:
        detector.stop_monitoring()

if __name__ == "__main__":
    main()
