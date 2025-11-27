#!/usr/bin/env python3
"""
Startup script for Interview Prep System Frontend

Launches the Streamlit app with proper configuration.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch Streamlit app."""
    # Get frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    app_file = frontend_dir / "interview_app.py"

    if not app_file.exists():
        print(f"Error: Frontend app not found at {app_file}")
        return 1

    print("=" * 60)
    print("Starting Interview Prep System - Visual Novel UI")
    print("=" * 60)
    print(f"\nApp location: {app_file}")
    print("\nLaunching Streamlit...")
    print("-" * 60)

    # Launch Streamlit
    try:
        subprocess.run([
            "streamlit", "run",
            str(app_file),
            "--theme.base", "dark",
            "--theme.primaryColor", "#ff6b9d",
            "--theme.backgroundColor", "#1a1625",
            "--theme.secondaryBackgroundColor", "#2d1b3d",
            "--theme.textColor", "#f0e6ff",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except FileNotFoundError:
        print("\nError: Streamlit is not installed!")
        print("Please install it with: pip install streamlit")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
