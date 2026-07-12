"""
Migration: Add Sprint 2 Verify pass columns to clips table.
Chay mot lan duy nhat khi deploy Sprint 3 build moi.
"""
from __future__ import annotations

import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "tools" / "emotion-data-studio"))

from backend.database.local_db import get_session


def run_migration() -> None:
    """Add verify columns to clips table if they don't exist."""
    session = get_session()
    try:
        # Check existing columns
        result = session.execute(text("PRAGMA table_info(clips)"))
        existing_cols = {row[1] for row in result.fetchall()}

        new_cols = {
            "verify_verdict": "TEXT",
            "verify_status": "TEXT DEFAULT 'not_run'",
            "verify_reasoning": "TEXT",
            "rejected_by_verify": "INTEGER DEFAULT 0",
        }

        for col, col_type in new_cols.items():
            if col not in existing_cols:
                sql = f"ALTER TABLE clips ADD COLUMN {col} {col_type}"
                session.execute(text(sql))
                print(f"  + Added column: {col} ({col_type})")
            else:
                print(f"  = Already exists: {col}")

        session.commit()
        print("\nMigration completed successfully.")

    except Exception as exc:
        session.rollback()
        print(f"Migration failed: {exc}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 Verify Columns Migration")
    print("=" * 60)
    run_migration()
