#!/usr/bin/env python3
"""Production server for LangGraph API on Render.

This script starts the LangGraph API server using the langgraph-cli
with proper configuration for production deployment.
"""
import os
import sys
import subprocess


def main():
    """Start the LangGraph API server with production settings."""
    # Get port from Render's environment variable, default to 10000
    port = os.environ.get("PORT", "10000")
    host = "0.0.0.0"

    print(f"Starting LangGraph API server on {host}:{port}")
    print(f"Environment: {'Production' if os.environ.get('RENDER') else 'Development'}")

    # Build the command to start LangGraph server
    # Using langgraph CLI to start the server
    cmd = [
        "langgraph",
        "up",
        "--host", host,
        "--port", port,
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run the server
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)


if __name__ == "__main__":
    main()
