#!/usr/bin/env python3
"""
Media Authenticity Detector - Enhanced Version
With logging, statistics, and improved error handling
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
import re
import os
from pathlib import Path

# Import config if available
try:
    from config import *
except ImportError:
    # Default values if config not found
    CHECK_INTERVAL = 10 
    NOTIFICATION_DURATION = 7000 
    AI_CONFIDENCE_THRESHOLD = 60
    MISLEADING_CONFIDENCE_THRESHOLD = 50
    LOG_DETECTIONS = True
    LOG_FILE_PATH = "detection_log.txt"
    RESIZE_SCREENSHOT_MAX = 1024
    JPEG_QUALITY = 85
    ANTHROPIC_API_KEY = None

# Get API key from environment variable or config
if 'ANTHROPIC_API_KEY' not in dir() or ANTHROPIC_API_KEY is None:
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
    print("Then set it using ONE of these methods:")
    print()
    print("Option 1 (Recommended): Environment Variable")
    print('  Windows CMD:         setx ANTHROPIC_API_KEY "your-key-here"')
    print('  Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key-here"')
    print('  Mac/Linux:           export ANTHROPIC_API_KEY="your-key-here"')
    print()
    print("Option 2: In config.py file")
    print("  Edit config.py and set:")
    print('  ANTHROPIC_API_KEY = "your-key-here"')
    print()
    print("After setting, restart this program.")
    print("=" * 60)
    exit(1)

class MediaDetectorEnhanced:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.monitoring = False
        self.last_check_time = 0
        self.check_interval = CHECK_INTERVAL
        self.notification_windows = []
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'ai_detected': 0,
            'misleading_detected': 0,
            'errors': 0,
            'start_time': None
        }
        
        # Logging
        self.log_enabled = LOG_DETECTIONS
        self.log_file = LOG_FILE_PATH
        
        # Initialize log file
        if self.log_enabled:
            self._init_log()
    
    def _init_log(self):
        """Initialize or create log file"""
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write(f"Media Detector Log - Started {datetime.now()}\n")
                    f.write("="*60 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
            self.log_enabled = False
    
    def _log(self, message, level="INFO"):
        """Log a message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        print(log_entry.strip())
        
        if self.log_enabled:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"Logging error: {e}")
    
    def capture_screen(self):
        """Capture the current screen with error handling"""
        try:
            screenshot = ImageGrab.grab()
            return screenshot
        except Exception as e:
            self._log(f"Error capturing screen: {e}", "ERROR")
            self.stats['errors'] += 1
            return None
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        try:
            buffered = BytesIO()
            image.thumbnail((RESIZE_SCREENSHOT_MAX, RESIZE_SCREENSHOT_MAX), Image.Resampling.LANCZOS)
            image.save(buffered, format="JPEG", quality=JPEG_QUALITY)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            self._log(f"Error converting image: {e}", "ERROR")
            return None
    
    def analyze_with_claude(self, image_base64):
        """Analyze image using Claude API with enhanced error handling"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages", 
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    # <--- CHANGED: Using the new Sonnet 4.5 model (Nov 2025)
                    "model": "claude-sonnet-4-5", 
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
                                    "text": """Analyze this screenshot for potential AI-generated content, **deepfakes**, or misinformation. 

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
- Check for **deepfake or AI artifacts**: Look for unnatural faces (e.g., waxy skin, odd eye reflections, mismatched features), weird hands, distorted backgrounds, inconsistent lighting, or garbled text.
- Pay close attention to images or videos of public figures, especially in political or sensational contexts, as these are common deepfake targets.
- Check if there's news content or claims that can be fact-checked
- Identify political content, propaganda, or sensational claims
- DO NOT flag professional graphics, UI elements, or legitimate content as AI

Be conservative - only flag content you're reasonably confident about."""
                                }
                            ]
                        }
                    ]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                result_text = data['content'][0]['text'].strip()
                result_text = re.sub(r'```json\s*', '', result_text)
                result_text = re.sub(r'```\s*', '', result_text)
                result_text = result_text.strip()
                
                result = json.loads(result_text)
                
                # Apply confidence thresholds
                if result.get('ai_confidence', 0) < AI_CONFIDENCE_THRESHOLD:
                    result['is_ai_generated'] = False
                if result.get('misleading_confidence', 0) < MISLEADING_CONFIDENCE_THRESHOLD:
                    result['is_misleading'] = False
                
                # Recalculate alert type based on thresholds
                ai_gen = result.get('is_ai_generated', False)
                mislead = result.get('is_misleading', False)
                if ai_gen and mislead:
                    result['alert_type'] = 'both'
                elif ai_gen:
                    result['alert_type'] = 'ai_generated'
                elif mislead:
                    result['alert_type'] = 'misleading'
                else:
                    result['alert_type'] = 'none'
                
                return result
            else:
                # Log the specific error from the API
                api_error_message = response.text
                try:
                    # Try to parse JSON for a cleaner error message
                    api_error_json = response.json()
                    if "error" in api_error_json and "message" in api_error_json["error"]:
                        api_error_message = api_error_json["error"]["message"]
                except:
                    pass # Keep the original text if not JSON
                
                self._log(f"API Error: {response.status_code} - {api_error_message[:200]}", "ERROR")
                self.stats['errors'] += 1
                return None
                
        except requests.Timeout:
            self._log("API request timeout", "WARNING")
            self.stats['errors'] += 1
            return None
        except json.JSONDecodeError as e:
            self._log(f"JSON parsing error: {e}. Response text: {result_text[:200]}...", "ERROR")
            self.stats['errors'] += 1
            return None
        except Exception as e:
            self._log(f"Error analyzing with Claude: {e}", "ERROR")
            self.stats['errors'] += 1
            return None
    
    def fact_check_claim(self, claim_text):
        """Fact-check a claim with error handling (OPTIMIZED to 1 call)"""
        try:
            # Fact check (Combined step)
            fact_check_response = requests.post(
                "https://api.anthropic.com/v1/messages", 
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    # <--- CHANGED: Using the new Haiku 4.5 model (Nov 2025)
                    "model": "claude-haiku-4-5",
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
                },
                timeout=20
            )
            
            if fact_check_response.status_code != 200:
                self._log(f"Fact-check (step 2) API Error: {fact_check_response.status_code}", "ERROR")
                return None
            
            fact_data = fact_check_response.json()
            fact_text = fact_data['content'][0]['text'].strip()
            fact_text = re.sub(r'```json\s*', '', fact_text)
            fact_text = re.sub(r'```\s*', '', fact_text)
            
            return json.loads(fact_text)
            
        except Exception as e:
            self._log(f"Error fact-checking: {e}", "WARNING")
            return None
    
    def get_context(self, claim_text):
        """Get detailed context with error handling"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages", 
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    # <--- CHANGED: Using the new Haiku 4.5 model (Nov 2025)
                    "model": "claude-haiku-4-5",
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
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text']
            
            self._log(f"Get Context API Error: {response.status_code}", "ERROR")
            return None
            
        except Exception as e:
            self._log(f"Error getting context: {e}", "WARNING")
            return None
    
    def show_notification(self, alert_type, message, claim_text=None):
        """Show notification with logging"""
        try:
            notification = tk.Toplevel()
            notification.withdraw()
            
            notification.overrideredirect(True)
            notification.attributes('-topmost', True)
            
            # Colors
            colors = {
                'ai_generated': "#FF6B6B",
                'misleading': "#FFA500",
                'both': "#DC143C"
            }
            
            titles = {
                'ai_generated': "⚠️ AI-Generated Content Detected",
                'misleading': "⚠️ Potentially Misleading Information",
                'both': "⚠️ AI-Generated & Misleading Content"
            }
            
            bg_color = colors.get(alert_type, "#FF6B6B")
            title_text = titles.get(alert_type, "⚠️ Alert")
            
            notification.configure(bg=bg_color)
            
            frame = tk.Frame(notification, bg=bg_color, padx=15, pady=10)
            frame.pack(fill=tk.BOTH, expand=True)
            
            title_label = tk.Label(
                frame, 
                text=title_text,
                font=('Arial', 12, 'bold'),
                bg=bg_color,
                fg='white'
            )
            title_label.pack(anchor='w')
            
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
            
            # Timestamp
            time_label = tk.Label(
                frame,
                text=f"Detected at: {datetime.now().strftime('%H:%M:%S')}",
                font=('Arial', 8),
                bg=bg_color,
                fg='white'
            )
            time_label.pack(anchor='w', pady=(3, 0))
            
            # Context button
            if claim_text and alert_type in ["misleading", "both"]:
                def show_context():
                    context_btn.config(state='disabled', text='Loading...')
                    notification.update()
                    
                    # Run context retrieval in a new thread to avoid blocking UI
                    def context_thread():
                        context = self.get_context(claim_text)
                        
                        def update_ui():
                            if context:
                                context_window = tk.Toplevel()
                                context_window.title("Context & Fact-Check")
                                context_window.geometry("500x400")
                                context_window.attributes('-topmost', True)
                                
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
                                
                                # Re-enable button on main notification
                                context_btn.config(state='normal', text='Show Context')
                            else:
                                context_btn.config(state='normal', text='Context (Failed)')
                        
                        # Schedule UI update back on the main thread
                        self.root.after(0, update_ui)
                        
                    threading.Thread(target=context_thread, daemon=True).start()

                context_btn = tk.Button(
                    frame,
                    text="Show Context",
                    command=show_context,
                    bg='white',
                    fg=bg_color,
                    font=('Arial', 9, 'bold'),
                    relief=tk.RAISED,
                    cursor='hand2'
                )
                context_btn.pack(anchor='w', pady=(8, 0))
            
            # Position
            notification.update_idletasks()
            width = notification.winfo_reqwidth()
            screen_width = notification.winfo_screenwidth()
            x = (screen_width - width) // 2
            y = 20
            
            notification.geometry(f"+{x}+{y}")
            notification.deiconify()
            
            def dismiss():
                try:
                    notification.destroy()
                    if notification in self.notification_windows:
                        self.notification_windows.remove(notification)
                except:
                    pass
            
            notification.after(NOTIFICATION_DURATION, dismiss)
            self.notification_windows.append(notification)
            
            # Log detection
            self.log(f"ALERT ({alert_type}): {message[:100]}", "DETECTION")
            
        except Exception as e:
            self._log(f"Error showing notification: {e}", "ERROR")
    
    def show_stats(self):
        """Display current statistics"""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        else:
            runtime_str = "Not started"
        
        stats_text = f"""
╔══════════════════════════════════════╗
║      Media Detector Statistics       ║
╠══════════════════════════════════════╣
║ Runtime: {runtime_str:25s} ║
║ Total Checks: {self.stats['total_checks']:20d} ║
║ AI Detected: {self.stats['ai_detected']:21d} ║
║ Misleading Detected: {self.stats['misleading_detected']:13d} ║
║ Errors: {self.stats['errors']:26d} ║
╚══════════════════════════════════════╝
"""
        print(stats_text)
    
    def monitoring_loop(self):
        """Enhanced monitoring loop with stats tracking"""
        while self.monitoring:
            current_time = time.time()
            
            if current_time - self.last_check_time >= self.check_interval:
                self.last_check_time = current_time
                self.stats['total_checks'] += 1
                
                screenshot = self.capture_screen()
                if screenshot:
                    self._log(f"Check #{self.stats['total_checks']}: Analyzing screen...", "INFO")
                    
                    image_base64 = self.image_to_base64(screenshot)
                    if image_base64:
                        analysis = self.analyze_with_claude(image_base64)
                        
                        if analysis:
                            alert_type = analysis.get('alert_type', 'none')
                            
                            if alert_type != 'none':
                                # Update stats
                                if analysis.get('is_ai_generated'):
                                    self.stats['ai_detected'] += 1
                                if analysis.get('is_misleading'):
                                    self.stats['misleading_detected'] += 1
                                
                                message = analysis.get('reason', 'Suspicious content detected')
                                claim_text = analysis.get('extracted_text', '')
                                
                                # Fact-check if needed (non-blocking)
                                if analysis.get('is_misleading') and claim_text:
                                    def fact_check_thread():
                                        fact_check = self.fact_check_claim(claim_text)
                                        if fact_check:
                                            verdict = fact_check.get('verdict', 'unverifiable')
                                            if verdict in ['false', 'misleading']:
                                                new_message = f"{message}\n\nFact-check: {fact_check.get('explanation', '')}"
                                                self.root.after(0, lambda: self.show_notification(
                                                    alert_type, new_message, claim_text
                                                ))
                                            else:
                                                # Show original notification if fact-check is not 'false'
                                                self.root.after(0, lambda: self.show_notification(
                                                    alert_type, message, claim_text
                                                ))
                                        else:
                                            # Show original notification if fact-check fails
                                            self.root.after(0, lambda: self.show_notification(
                                                alert_type, message, claim_text
                                            ))
                                    
                                    # Start fact-check in a separate thread
                                    threading.Thread(target=fact_check_thread, daemon=True).start()
                                else:
                                    # Show notification immediately if not misleading
                                    self.root.after(0, lambda: self.show_notification(
                                        alert_type, 
                                        message,
                                        None # No claim text to pass
                                    ))
                            else:
                                self._log("No suspicious content detected", "INFO")
            
            time.sleep(0.5)
    
    def start_monitoring(self):
        """Start monitoring with enhanced feedback"""
        if not self.monitoring:
            self.monitoring = True
            self.stats['start_time'] = datetime.now()
            
            print("\n" + "="*60)
            print("🛡️  MEDIA AUTHENTICITY DETECTOR - ENHANCED VERSION")
            print("="*60)
            print(f"Started: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Check interval: {self.check_interval} seconds")
            print(f"Logging: {'Enabled' if self.log_enabled else 'Disabled'}")
            if self.log_enabled:
                print(f"Log file: {self.log_file}")
            print("\n🔍 Monitoring your screen for AI-generated and misleading content...")
            print("📊 Press Ctrl+C to stop and see statistics\n")
            print("="*60 + "\n")
            
            self._log("Media Detector started", "INFO")
            
            monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            monitor_thread.start()
            
            self.root.mainloop()
    
    def stop_monitoring(self):
        """Stop monitoring with statistics"""
        self.monitoring = False
        print("\n" + "="*60)
        print("Stopping Media Detector...")
        print("="*60)
        self._log("Media Detector stopped", "INFO")
        self.show_stats()
        print("\n✅ Media Detector stopped successfully.")
        print("="*60 + "\n")

def main():
    print("\n🚀 Initializing Media Authenticity Detector...")
    detector = MediaDetectorEnhanced()
    
    try:
        detector.start_monitoring()
    except KeyboardInterrupt:
        detector.stop_monitoring()
    finally:
        # Ensure the root window is destroyed on exit
        if detector.root:
            try:
                detector.root.destroy()
            except tk.TclError:
                pass # Window might already be destroyed

if __name__ == "__main__":
    main()