"""
Emotion Data Studio — Theme Constants
======================================
Centralized color palette, spacing, typography, and animation constants.
Used by both QSS and Python widget code for consistent styling.
"""


class Colors:
    """Color palette — Deep dark + Purple accent"""

    # === Backgrounds ===
    BG_DARKEST = "#06060a"        # Deepest background
    BG_PRIMARY = "#0a0a0f"        # Main window background
    BG_SECONDARY = "#0f0f17"      # Cards / panels
    BG_TERTIARY = "#14141f"       # Elevated cards
    BG_HOVER = "#1a1a28"          # Hover state
    BG_ACTIVE = "#1f1f30"         # Active/selected state
    BG_INPUT = "#0d0d15"          # Input fields

    # === Sidebar ===
    SIDEBAR_BG = "#08080d"        # Sidebar background
    SIDEBAR_HOVER = "#12121c"     # Sidebar item hover
    SIDEBAR_ACTIVE = "#1a1a2e"    # Sidebar item active

    # === Accent (Purple) ===
    ACCENT_PRIMARY = "#6c5ce7"    # Primary purple
    ACCENT_LIGHT = "#a29bfe"      # Light purple (hover)
    ACCENT_DARK = "#4a3db8"       # Dark purple (pressed)
    ACCENT_GLOW = "rgba(108, 92, 231, 0.3)"   # Glow effect
    ACCENT_SUBTLE = "rgba(108, 92, 231, 0.08)" # Subtle bg tint

    # === Text ===
    TEXT_PRIMARY = "#e8e6f0"      # Primary text (slightly purple-tinted white)
    TEXT_SECONDARY = "#9896a8"    # Secondary text
    TEXT_TERTIARY = "#5c5a6e"     # Muted text
    TEXT_INVERSE = "#0a0a0f"      # Text on light background

    # === Borders ===
    BORDER_SUBTLE = "rgba(255, 255, 255, 0.06)"   # Very subtle border
    BORDER_DEFAULT = "rgba(255, 255, 255, 0.10)"   # Default border
    BORDER_STRONG = "rgba(255, 255, 255, 0.15)"    # Strong border
    BORDER_ACCENT = "rgba(108, 92, 231, 0.4)"      # Accent border

    # === Status Colors ===
    SUCCESS = "#00b894"           # Green — approved, completed
    SUCCESS_BG = "rgba(0, 184, 148, 0.12)"
    WARNING = "#fdcb6e"           # Yellow — pending, needs review
    WARNING_BG = "rgba(253, 203, 110, 0.12)"
    ERROR = "#e17055"             # Red/Orange — rejected, error
    ERROR_BG = "rgba(225, 112, 85, 0.12)"
    INFO = "#74b9ff"              # Blue — info, processing
    INFO_BG = "rgba(116, 185, 255, 0.12)"

    # === Emotion Colors ===
    EMOTION_HAPPY = "#00b894"
    EMOTION_SAD = "#74b9ff"
    EMOTION_ANGRY = "#e17055"
    EMOTION_FEAR = "#a29bfe"
    EMOTION_SURPRISE = "#fdcb6e"
    EMOTION_DISGUST = "#e84393"
    EMOTION_NEUTRAL = "#636e72"

    # === Glassmorphism ===
    GLASS_BG = "rgba(14, 14, 22, 0.75)"
    GLASS_BORDER = "rgba(255, 255, 255, 0.08)"
    GLASS_SHADOW = "rgba(0, 0, 0, 0.5)"


class Spacing:
    """Spacing system (multiples of 4px)"""
    XXS = 2
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32
    XXXL = 48


class BorderRadius:
    """Border radius values"""
    SM = 6
    MD = 8
    LG = 12
    XL = 16
    ROUND = 999   # Pill shape


class Typography:
    """Font settings"""
    FAMILY = "Inter, Segoe UI, sans-serif"
    FAMILY_MONO = "JetBrains Mono, Cascadia Code, Consolas, monospace"

    # Font sizes
    SIZE_XS = 10
    SIZE_SM = 11
    SIZE_MD = 13
    SIZE_LG = 15
    SIZE_XL = 18
    SIZE_XXL = 22
    SIZE_DISPLAY = 28

    # Font weights
    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700


class Animation:
    """Animation durations (ms)"""
    FAST = 150
    NORMAL = 250
    SLOW = 400
    VERY_SLOW = 600


class Sizes:
    """Component sizes"""
    SIDEBAR_WIDTH_COLLAPSED = 64
    SIDEBAR_WIDTH_EXPANDED = 220
    SIDEBAR_ICON_SIZE = 20
    TITLEBAR_HEIGHT = 38
    STATUSBAR_HEIGHT = 28
    CARD_MIN_HEIGHT = 80
    BUTTON_HEIGHT = 36
    BUTTON_HEIGHT_SM = 28
    INPUT_HEIGHT = 36
    SCROLLBAR_WIDTH = 8


# Emotion label mapping
EMOTION_MAP = {
    "happy": {"label": "Vui vẻ", "emoji": "😊", "color": Colors.EMOTION_HAPPY, "shortcut": "1"},
    "sad": {"label": "Buồn", "emoji": "😢", "color": Colors.EMOTION_SAD, "shortcut": "2"},
    "angry": {"label": "Giận dữ", "emoji": "😠", "color": Colors.EMOTION_ANGRY, "shortcut": "3"},
    "fear": {"label": "Sợ hãi", "emoji": "😨", "color": Colors.EMOTION_FEAR, "shortcut": "4"},
    "surprise": {"label": "Ngạc nhiên", "emoji": "😲", "color": Colors.EMOTION_SURPRISE, "shortcut": "5"},
    "disgust": {"label": "Ghê tởm", "emoji": "🤢", "color": Colors.EMOTION_DISGUST, "shortcut": "6"},
    "neutral": {"label": "Trung tính", "emoji": "😐", "color": Colors.EMOTION_NEUTRAL, "shortcut": "7"},
}
