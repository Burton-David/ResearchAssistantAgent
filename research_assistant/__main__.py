#!/usr/bin/env python3
"""
Main entry point for Research Assistant.

Allows running as python -m research_assistant
"""

from .cli_enhanced import cli

if __name__ == "__main__":
    cli()