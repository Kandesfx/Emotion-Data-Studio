"""
Emotion Data Studio — Cloud SQL Client
========================================
Connect to Google Cloud SQL (PostgreSQL) for cloud-side data.
Used for multi-device sync and team collaboration.

Connection methods:
  1. Cloud SQL Auth Proxy (recommended for dev)
  2. Direct TCP (for Cloud Run + VPC connector)
  3. Unix socket (for Cloud Run with built-in connector)
"""

import os
import logging
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)


class CloudSQLClient:
    """
    Google Cloud SQL (PostgreSQL) connection manager.
    Mirrors the local SQLite schema for seamless sync.
    """

    def __init__(self):
        from backend.config import settings
        self.connection_name = settings.CLOUD_SQL_CONNECTION_NAME
        self.db_user = settings.CLOUD_SQL_USER
        self.db_password = settings.CLOUD_SQL_PASSWORD
        self.db_name = settings.CLOUD_SQL_DB
        self.project_id = settings.GCP_PROJECT_ID
        self.credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS

        self._engine = None
        self._session_factory = None

    @property
    def is_configured(self) -> bool:
        """Check if Cloud SQL is properly configured"""
        return all([
            self.connection_name,
            self.db_user,
            self.db_password,
            self.db_name,
        ])

    def _get_engine(self):
        """Create SQLAlchemy engine for Cloud SQL"""
        if self._engine is None:
            if not self.is_configured:
                raise ValueError("Cloud SQL not configured. Set environment variables.")

            # Try different connection methods
            engine = self._try_cloud_run_connector()
            if engine is None:
                engine = self._try_proxy_connection()
            if engine is None:
                engine = self._try_direct_connection()

            if engine is None:
                raise ConnectionError("Could not connect to Cloud SQL via any method")

            self._engine = engine
            self._session_factory = sessionmaker(bind=engine)

            # Create tables if they don't exist
            from backend.database.local_db import Base
            Base.metadata.create_all(bind=engine)

            logger.info("Cloud SQL engine initialized")

        return self._engine

    def _try_cloud_run_connector(self):
        """Method 1: Cloud SQL Python Connector (best for Cloud Run)"""
        try:
            from google.cloud.sql.connector import Connector

            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

            connector = Connector()

            def getconn():
                return connector.connect(
                    self.connection_name,
                    "pg8000",
                    user=self.db_user,
                    password=self.db_password,
                    db=self.db_name,
                )

            engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn,
                pool_size=5,
                max_overflow=2,
                pool_timeout=30,
                pool_recycle=1800,
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected via Cloud SQL Python Connector")
            return engine

        except ImportError:
            logger.debug("cloud-sql-python-connector not installed")
            return None
        except Exception as e:
            logger.debug(f"Cloud Run connector failed: {e}")
            return None

    def _try_proxy_connection(self):
        """Method 2: Cloud SQL Auth Proxy (localhost:5432)"""
        try:
            db_url = (
                f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
                f"@127.0.0.1:5432/{self.db_name}"
            )
            engine = create_engine(db_url, pool_size=5, pool_timeout=10)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected via Cloud SQL Auth Proxy")
            return engine

        except Exception as e:
            logger.debug(f"Proxy connection failed: {e}")
            return None

    def _try_direct_connection(self):
        """Method 3: Direct TCP (needs VPC / public IP)"""
        try:
            # Extract host from connection name
            # Format: project:region:instance
            parts = self.connection_name.split(":")
            if len(parts) == 3:
                # Try the instance name as host (requires public IP)
                host = os.getenv("CLOUD_SQL_HOST", "127.0.0.1")
                port = os.getenv("CLOUD_SQL_PORT", "5432")

                db_url = (
                    f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
                    f"@{host}:{port}/{self.db_name}"
                )
                engine = create_engine(db_url, pool_size=5, pool_timeout=10)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(f"Connected via direct TCP: {host}:{port}")
                return engine

        except Exception as e:
            logger.debug(f"Direct connection failed: {e}")
            return None

    # ================================================================
    # SESSION MANAGEMENT
    # ================================================================

    @contextmanager
    def get_session(self):
        """
        Get a Cloud SQL session (context manager).
        
        Usage:
            with cloud_sql.get_session() as session:
                session.query(Video).all()
        """
        self._get_engine()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_raw_session(self) -> Session:
        """Get a raw session (caller manages lifecycle)"""
        self._get_engine()
        return self._session_factory()

    # ================================================================
    # HEALTH CHECK
    # ================================================================

    def test_connection(self) -> dict:
        """
        Test Cloud SQL connection and return status.
        
        Returns:
            dict with status, message, latency_ms
        """
        import time

        if not self.is_configured:
            return {
                "status": "not_configured",
                "message": "Cloud SQL credentials not set",
                "latency_ms": 0,
            }

        try:
            start = time.time()
            self._get_engine()
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            latency = (time.time() - start) * 1000

            return {
                "status": "connected",
                "message": f"Connected to {self.connection_name}",
                "latency_ms": round(latency, 1),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "latency_ms": 0,
            }
