# -*- mode: python ; coding: utf-8 -*-
"""
Emotion Data Studio — PyInstaller Spec File
=============================================
Build command: pyinstaller build/emotion_studio.spec
Output: dist/EmotionDataStudio/
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'app.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # UI styles & resources
        (os.path.join(PROJECT_ROOT, 'ui', 'styles', 'dark_theme.qss'), os.path.join('ui', 'styles')),
        (os.path.join(PROJECT_ROOT, 'ui', 'styles', 'theme.py'), os.path.join('ui', 'styles')),
        # Backend source (needed for imports)
        (os.path.join(PROJECT_ROOT, 'backend'), 'backend'),
    ],
    hiddenimports=[
        # PySide6 modules
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        # SQLAlchemy dialects
        'sqlalchemy.dialects.sqlite',
        # Backend modules
        'backend.config',
        'backend.database',
        'backend.database.local_db',
        'backend.database.models',
        'backend.services',
        'backend.services.downloader',
        'backend.services.scene_splitter',
        'backend.services.face_extractor',
        'backend.services.audio_extractor',
        'backend.services.transcriber',
        'backend.services.emotion_analyzer',
        'backend.services.quality_scorer',
        'backend.services.pipeline_orchestrator',
        'backend.ai_models',
        'backend.ai_models.model_manager',
        # UI modules
        'ui',
        'ui.main_window',
        'ui.pages',
        'ui.pages.dashboard_page',
        'ui.pages.processing_page',
        'ui.pages.review_page',
        'ui.pages.export_page',
        'ui.widgets',
        'ui.widgets.sidebar',
        'ui.workers',
        'ui.workers.pipeline_worker',
        'ui.workers.export_worker',
        'ui.styles',
        'ui.styles.theme',
        # Pydantic
        'pydantic',
        'pydantic_settings',
        # ML libraries (optional — loaded at runtime)
        'sklearn',
        'sklearn.model_selection',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML packages from base build
        # They will be loaded dynamically at runtime if available
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'whisper',
        'deepface',
        'insightface',
        'mediapipe',
        # Exclude server-only packages
        'fastapi',
        'uvicorn',
        'flask',
        'celery',
        'redis',
        # Exclude unnecessary packages
        'tensorboard',
        'jupyter',
        'notebook',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EmotionDataStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, 'assets', 'icon.ico') if os.path.exists(os.path.join(PROJECT_ROOT, 'assets', 'icon.ico')) else None,
    version_info=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EmotionDataStudio',
)
