import sys
import os

# Get the path to the src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add src path to sys.path so module2 can be imported
if src_path not in sys.path:
    sys.path.append(src_path)