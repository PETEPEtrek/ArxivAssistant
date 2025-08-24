#!/usr/bin/env python3

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ui.app import ArxivAssistantApp

if __name__ == "__main__":
    app = ArxivAssistantApp()
    app.run()
