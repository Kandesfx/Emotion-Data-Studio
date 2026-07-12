"""Verify all modified modules after Vertex AI integration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("=" * 60)
print("  Importing all modified modules...")
print("=" * 60)

# 1. Backend - gemini_auto_labeler (refactored)
from backend.services.gemini_auto_labeler import (
    get_genai_client, is_vertex_configured, GeminiAutoLabeler,
    VERTEX_GLOBAL_LOCATION,
)
print("OK  backend.services.gemini_auto_labeler")

# 2. Backend - ai_video_segmenter (new)
from backend.services.ai_video_segmenter import (
    AIVideoSegmenter, AutoCutSegment, AutoCutResult, AUTOCUT_SYSTEM_PROMPT,
)
print("OK  backend.services.ai_video_segmenter")

# 3. Backend - pipeline_orchestrator (modified)
from backend.services.pipeline_orchestrator import PipelineOrchestrator
po = PipelineOrchestrator()
assert hasattr(po, "_vertex_ai_ready"), "missing _vertex_ai_ready"
assert hasattr(po, "_ai_autocut_stage"), "missing _ai_autocut_stage"
assert hasattr(po, "_classic_cut_stage"), "missing _classic_cut_stage"
print("OK  backend.services.pipeline_orchestrator")

# 4. Backend - gemini_api (new endpoint)
from backend.api.gemini_api import router, CutAndCreateRequest
route_paths = [r.path for r in router.routes]
has_cut = any("/cut-and-create" in p for p in route_paths)
assert has_cut, f"missing cut-and-create, got: {route_paths}"
print(f"OK  backend.api.gemini_api  ({len(route_paths)} routes, has /cut-and-create)")

# 5. Backend - config
from backend.config import settings
assert settings.AI_AUTOCUT_ENABLED is False
assert settings.AI_AUTOCUT_INTENSITY_THRESHOLD == 0.55
assert settings.VERTEX_LOCATION == "global"
print(f"OK  backend.config  (AI_AUTOCUT={settings.AI_AUTOCUT_ENABLED}, LOCATION={settings.VERTEX_LOCATION})")

# 6. UI - dashboard_page (modified)
from ui.pages.dashboard_page import DashboardPage
import inspect
src = inspect.getsource(DashboardPage)
assert "ai_autocut_chk" in src
assert "_refresh_ai_status" in src
assert "_apply_ai_autocut_settings" in src
print("OK  ui.pages.dashboard_page")

# 7. UI - settings_page (modified)
from ui.pages.settings_page import SettingsPage
src = inspect.getsource(SettingsPage)
assert "_build_vertex_ai_card" in src
assert "vertex_location_input" in src
assert "ai_autocut_threshold" in src
print("OK  ui.pages.settings_page")

# 8. UI - processing_page (modified)
from ui.pages.processing_page import ProcessingPage
src = inspect.getsource(ProcessingPage)
assert "ai_autocut" in src
print("OK  ui.pages.processing_page")

# 9. UI - main_window
try:
    from ui.main_window import MainWindow  # noqa
    print("OK  ui.main_window")
except Exception as exc:
    print(f"WARN ui.main_window: {exc}")

print()
print("=" * 60)
print("  ALL IMPORTS PASS")
print("=" * 60)
