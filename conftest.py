# backend/conftest.py
import sys
import os

# This file is apparently automatically discovered and used by pytest - no need to manually import or call.
# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
