"""Uvicorn entry point for the local authenticated API."""

from local_app.api import app, create_app

__all__ = ["app", "create_app"]
