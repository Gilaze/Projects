# Media Detector Configuration File

# =============================================================================
# API CONFIGURATION - IMPORTANT!
# =============================================================================
# You need a Claude API key to use this tool.
# Get your API key from: https://console.anthropic.com/
# 
# Option 1 (Recommended): Set as environment variable
#   Windows: setx ANTHROPIC_API_KEY "your-api-key-here"
#   Mac/Linux: export ANTHROPIC_API_KEY="your-api-key-here"
#
# Option 2: Uncomment and add your key below (less secure)
# ANTHROPIC_API_KEY = "sk-ant-api03-..."

ANTHROPIC_API_KEY = None  # Will use environment variable if this is None

# =============================================================================
# MONITORING SETTINGS
# =============================================================================
CHECK_INTERVAL = 3  # Seconds between each screen check (lower = more frequent, higher CPU usage)
AUTO_START = True  # Start monitoring immediately on launch

# =============================================================================
# NOTIFICATION SETTINGS
# =============================================================================
NOTIFICATION_DURATION = 5000  # Milliseconds (5000 = 5 seconds)
NOTIFICATION_POSITION = "top_center"  # Options: top_center, top_right, top_left
SHOW_TIMESTAMP = True  # Show time when content was detected

# =============================================================================
# DETECTION SENSITIVITY
# =============================================================================
AI_CONFIDENCE_THRESHOLD = 60  # Minimum confidence (0-100) to show AI alert
MISLEADING_CONFIDENCE_THRESHOLD = 50  # Minimum confidence to show misleading alert

# =============================================================================
# ALERT COLORS (hex codes)
# =============================================================================
AI_ALERT_COLOR = "#FF6B6B"  # Red
MISLEADING_ALERT_COLOR = "#FFA500"  # Orange
BOTH_ALERT_COLOR = "#DC143C"  # Crimson

# =============================================================================
# ADVANCED FEATURES
# =============================================================================
ENABLE_SOUND_ALERTS = False  # Play sound when content detected (requires additional setup)
LOG_DETECTIONS = True  # Save detection log to file
LOG_FILE_PATH = "detection_log.txt"

# =============================================================================
# PERFORMANCE
# =============================================================================
RESIZE_SCREENSHOT_MAX = 1024  # Max dimension for screenshot before analysis (lower = faster, less accurate)
JPEG_QUALITY = 85  # Screenshot quality (0-100, higher = better quality, more tokens)

# =============================================================================
# FACT-CHECKING
# =============================================================================
ENABLE_FACT_CHECK = True  # Perform fact-checking on text content
MAX_SEARCH_QUERIES = 2  # Number of search queries for fact-checking

# =============================================================================
# UI CUSTOMIZATION
# =============================================================================
NOTIFICATION_OPACITY = 1.0  # 0.0 (transparent) to 1.0 (opaque)
FONT_SIZE_TITLE = 12
FONT_SIZE_MESSAGE = 10
NOTIFICATION_WIDTH = 400  # pixels
