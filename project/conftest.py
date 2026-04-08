"""Configure sys.path so tests can import the env package."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
