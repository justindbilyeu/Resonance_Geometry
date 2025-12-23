#!/usr/bin/env python3
"""
F_AI Score - Command-line entrypoint script

Thin wrapper around src.f_ai.cli.main()
"""

import sys
from pathlib import Path

# Add src to path if needed
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.f_ai.cli import main

if __name__ == "__main__":
    main()
