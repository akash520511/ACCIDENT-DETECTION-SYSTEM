# config.py - Complete Configuration File
import os
import cv2  # ← ADD THIS IMPORT
from pathlib import Path

# ============================================
# BASE PATHS
# ============================================
BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEST_VIDEOS_FOLDER = os.path.join(BASE_DIR, 'test_videos')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
EVIDENCE_FOLDER = os.path.join(BASE_DIR, 'evidence')
DATABASE_FOLDER = os.path.join(BASE_DIR, 'database')

# Create all folders
for folder in [UPLOAD_FOLDER, TEST_VIDEOS_FOLDER, PROCESSED_FOLDER,
               EVIDENCE_FOLDER, DATABASE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ============================================
# VIDEO SETTINGS
# ============================================
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB

# Camera/Display settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FPS_TARGET = 30

# ============================================
# DETECTION SETTINGS
# ============================================
ACCIDENT_THRESHOLD = 0.6
MIN_VEHICLES_FOR_ACCIDENT = 2
OVERLAP_THRESHOLD = 500  # Pixel overlap to detect collision

# Vehicle detection classes (YOLO)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# ============================================
# ACCIDENT CONFIRMATION SETTINGS (SINGLE ACCIDENT)
# ============================================
ACCIDENT_CONFIRMATION_FRAMES = 5  # Number of consecutive frames needed to confirm
ACCIDENT_REPORT_COOLDOWN = 10  # Seconds to wait before reporting another accident
MULTI_FRAME_CONFIRMATION = 5  # Frames needed for confirmation (same as above)
DUPLICATE_ALERT_PREVENTION = True  # Prevent duplicate alerts

# ============================================
# SEVERITY THRESHOLDS
# ============================================
SEVERITY_THRESHOLDS = {
    'MINOR': 30,  # 0-30% = Minor accident
    'MAJOR': 60,  # 30-60% = Major accident
    'CRITICAL': 90  # 60-100% = Critical accident
}

# Severity scoring weights
SEVERITY_WEIGHTS = {
    'vehicle_count': 30,  # Max 30 points
    'overlap': 25,  # Max 25 points
    'motion': 20,  # Max 20 points
    'debris': 15,  # Max 15 points
    'speed_change': 10  # Max 10 points
}

# ============================================
# CONFIDENCE THRESHOLDS
# ============================================
CONFIDENCE_LEVELS = {
    'HIGH': 85,  # 85-100% = High confidence (Green)
    'MEDIUM': 65,  # 65-85% = Medium confidence (Yellow)
    'LOW': 0  # 0-65% = Low confidence (Red)
}

# Confidence scoring weights
CONFIDENCE_WEIGHTS = {
    'detection': 40,  # Detection confidence (0-40)
    'severity': 30,  # Severity confidence (0-30)
    'temporal': 20,  # Temporal consistency (0-20)
    'scene_context': 10  # Scene context (0-10)
}

# ============================================
# ALERT SETTINGS
# ============================================
ALERT_COOLDOWN = 5  # Seconds between alerts
SAVE_EVIDENCE = True  # Save evidence on accident
EMERGENCY_COOLDOWN = 5  # Cooldown for emergency alerts

# Alert channels (enable/disable)
SMS_ENABLED = True
EMAIL_ENABLED = True
TELEGRAM_ENABLED = False
WHATSAPP_ENABLED = False
BUZZER_ENABLED = False
DASHBOARD_NOTIFICATIONS = True

# Emergency contact numbers
EMERGENCY_CONTACTS = {
    'police': '100',
    'ambulance': '108',
    'fire': '101'
}

# Demo contacts (for testing)
DEMO_PHONE_NUMBER = "+1234567890"
DEMO_EMAIL = "family@example.com"

# Email configuration (for production)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "accident.detection.system@gmail.com"
SENDER_PASSWORD = "your-app-password-here"  # Use environment variable in production

# ============================================
# ANPR (License Plate Recognition) SETTINGS
# ============================================
ANPR_ENABLED = True
ANPR_CONFIDENCE_THRESHOLD = 0.7
ANPR_MIN_PLATE_LENGTH = 4
ANPR_MAX_PLATE_LENGTH = 12

# License plate patterns (Indian format)
LICENSE_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',  # TN01AB1234
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # TN01AB1234
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',  # TN01A1234
    r'^[A-Z]{2}[0-9]{2}[0-9]{4}$',  # TN011234
    r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',  # 01AB1234
]

# ============================================
# COLORS (BGR format for OpenCV)
# ============================================
COLORS = {
    # Severity colors
    'MINOR': (0, 255, 255),  # Yellow
    'MAJOR': (0, 165, 255),  # Orange
    'CRITICAL': (0, 0, 255),  # Red
    'NORMAL': (0, 255, 0),  # Green

    # Confidence colors
    'HIGH_CONF': (0, 255, 0),  # Green
    'MED_CONF': (0, 255, 255),  # Yellow
    'LOW_CONF': (0, 0, 255),  # Red

    # Basic colors
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (128, 128, 128),
    'RED': (0, 0, 255),
    'GREEN': (0, 255, 0),
    'BLUE': (255, 0, 0),
    'YELLOW': (0, 255, 255),
    'CYAN': (255, 255, 0),
    'MAGENTA': (255, 0, 255),

    # Detection box colors
    'VEHICLE_BOX': (0, 255, 0),
    'ACCIDENT_BOX': (0, 0, 255),
    'PLATE_BOX': (255, 0, 0)
}

# ============================================
# UI DASHBOARD SETTINGS
# ============================================
DASHBOARD_WIDTH = 350  # Sidebar width in pixels
SHOW_VEHICLE_COUNT = True
SHOW_CONFIDENCE = True
SHOW_SEVERITY = True
SHOW_PERFORMANCE = True
SHOW_TREND_GRAPH = True
SHOW_CONTROLS = True

# Font settings (cv2 is now imported)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.5
FONT_SCALE_MEDIUM = 0.6
FONT_SCALE_LARGE = 0.8
FONT_THICKNESS = 1
FONT_THICKNESS_BOLD = 2

# ============================================
# PERFORMANCE METRICS (as per PPT)
# ============================================
TARGET_ACCURACY = 94.2  # Target accuracy from PPT
TARGET_RESPONSE_TIME = 1.3  # Target response time in seconds
TARGET_PRECISION = 89.5  # Target precision
TARGET_RECALL = 87.3  # Target recall

# Confusion matrix values (from PPT)
CONFUSION_MATRIX = {
    'true_positives': 235,
    'true_negatives': 236,
    'false_positives': 14,
    'false_negatives': 15
}

# ============================================
# EVIDENCE CAPTURE SETTINGS
# ============================================
EVIDENCE_FORMAT = 'jpg'
EVIDENCE_QUALITY = 95  # JPEG quality (1-100)
SAVE_ANNOTATED_FRAMES = True
SAVE_HEATMAP = True
SAVE_METADATA = True
MAX_EVIDENCE_FILES = 1000  # Maximum evidence files to keep

# Video recording settings
RECORDING_FPS = 20
RECORDING_FOURCC = 'XVID'
RECORDING_FORMAT = 'avi'

# ============================================
# DATABASE SETTINGS
# ============================================
DATABASE_BACKUP_ENABLED = True
DATABASE_BACKUP_INTERVAL = 3600  # Backup every hour
MAX_DATABASE_RECORDS = 1000  # Keep last 1000 records

# ============================================
# LOGGING SETTINGS
# ============================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "system.log"
LOG_TO_CONSOLE = True
LOG_TO_FILE = True
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# ============================================
# WEB INTERFACE SETTINGS (for Flask app)
# ============================================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
SECRET_KEY = 'your-secret-key-here-change-in-production'
SESSION_TIMEOUT = 3600  # Session timeout in seconds

# ============================================
# HEATMAP VISUALIZATION SETTINGS
# ============================================
HEATMAP_ALPHA = 0.6  # Transparency (0-1)
HEATMAP_COLORMAP = 'JET'  # JET, HOT, COOL, etc.
HEATMAP_INTENSITY_MAX = 1.0
GAUSSIAN_SIGMA_FACTOR = 3

# ============================================
# OPTICAL FLOW SETTINGS
# ============================================
OPTICAL_FLOW_PARAMS = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

MOTION_THRESHOLD = 20  # Threshold for significant motion
HIGH_MOTION_THRESHOLD = 40  # Threshold for high motion

# ============================================
# DEBRIS DETECTION SETTINGS
# ============================================
DEBRIS_CANNY_LOW = 50
DEBRIS_CANNY_HIGH = 150
DEBRIS_AREA_EXPANSION = 50  # Pixels to expand ROI
DEBRIS_EDGE_DENSITY_MAX = 20

# ============================================
# SPEED ESTIMATION SETTINGS
# ============================================
PIXELS_PER_METER = 10  # Approximate calibration
FPS_ESTIMATION = 30
VEHICLE_SPEED_THRESHOLD = 50  # km/h for sudden stop detection

# ============================================
# TESTING & DEMO SETTINGS
# ============================================
DEMO_MODE_ENABLED = True
SIMULATE_VEHICLES = False  # Set to False to use real YOLO detection
SIMULATED_VEHICLE_COUNT = 3
SIMULATED_ACCIDENT_PROBABILITY = 0.3

# ============================================
# EXPORT SETTINGS
# ============================================
EXPORT_FORMATS = ['json', 'csv', 'html']
REPORT_TEMPLATE = 'report_template.html'
AUTO_EXPORT_ON_ACCIDENT = False

# ============================================
# SYSTEM LIMITS
# ============================================
MAX_FRAME_QUEUE_SIZE = 10
MAX_HISTORY_SIZE = 100
MAX_RESPONSE_TIMES = 1000
MAX_PROCESSING_THREADS = 4

# ============================================
# API KEYS (for production - use environment variables)
# ============================================
# Twilio for SMS
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')

# SendGrid for Email
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY', '')

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_confidence_color(score):
    """Get color based on confidence score"""
    if score >= CONFIDENCE_LEVELS['HIGH']:
        return COLORS['HIGH_CONF']
    elif score >= CONFIDENCE_LEVELS['MEDIUM']:
        return COLORS['MED_CONF']
    else:
        return COLORS['LOW_CONF']


def get_severity_color(severity):
    """Get color based on severity level"""
    colors = {
        'MINOR': COLORS['MINOR'],
        'MAJOR': COLORS['MAJOR'],
        'CRITICAL': COLORS['CRITICAL'],
        'NONE': COLORS['NORMAL']
    }
    return colors.get(severity, COLORS['WHITE'])


def get_severity_level(score):
    """Get severity level based on score"""
    if score >= SEVERITY_THRESHOLDS['CRITICAL']:
        return 'CRITICAL'
    elif score >= SEVERITY_THRESHOLDS['MAJOR']:
        return 'MAJOR'
    elif score >= SEVERITY_THRESHOLDS['MINOR']:
        return 'MINOR'
    else:
        return 'NONE'


def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check folders
    for folder in [UPLOAD_FOLDER, TEST_VIDEOS_FOLDER, PROCESSED_FOLDER,
                   EVIDENCE_FOLDER, DATABASE_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Check thresholds
    if ACCIDENT_CONFIRMATION_FRAMES < 1:
        errors.append("ACCIDENT_CONFIRMATION_FRAMES must be at least 1")

    if ACCIDENT_REPORT_COOLDOWN < 1:
        errors.append("ACCIDENT_REPORT_COOLDOWN must be at least 1")

    if OVERLAP_THRESHOLD < 0:
        errors.append("OVERLAP_THRESHOLD must be positive")

    if errors:
        print("⚠️ Configuration warnings:")
        for error in errors:
            print(f"   - {error}")
        return False

    return True


# Run validation on import
if __name__ != "__main__":
    validate_config()

print("✅ Configuration loaded successfully")