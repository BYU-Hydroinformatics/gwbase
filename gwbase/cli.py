"""
GWBASE Command Line Interface

This module provides the CLI entry point for the gwbase package.
"""

import sys
from pathlib import Path

# Add parent directory to path to import main_gwbase
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main_gwbase import main


if __name__ == '__main__':
    main()
