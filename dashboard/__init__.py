"""Gradio dashboard for Training Dynamics Workbench.

This package provides a web interface for:
- Configuring and launching model training (REQ_007)
- Running analysis and viewing synchronized visualizations (REQ_008)
- Loss curves with epoch indicator (REQ_009)
- Application versioning (REQ_010)

Usage:
    python -m dashboard.app
"""

from dashboard.app import create_app
from dashboard_v2.version import __version__

__all__ = ["create_app", "__version__"]
