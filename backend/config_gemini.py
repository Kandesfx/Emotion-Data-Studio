# ── Google Cloud / Gemini Auto-Labeler ──────────────────────────────────────

# Vertex AI settings
GCP_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID", None)
GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", None
)

# Gemini Auto-Labeler
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
GEMINI_INTENSITY_THRESHOLD: float = float(
    os.getenv("GEMINI_INTENSITY_THRESHOLD", "0.6")
)

# Gemini Budget Control
GEMINI_MONTHLY_BUDGET_USD: float = float(
    os.getenv("GEMINI_MONTHLY_BUDGET_USD", "500.0")
)
GEMINI_COST_TRACKING_ENABLED: bool = (
    os.getenv("GEMINI_COST_TRACKING_ENABLED", "true").strip().lower() == "true"
)
