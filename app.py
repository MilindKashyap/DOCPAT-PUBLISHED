"""
Temporary app.py file to work around Render's default start command.
This imports the Django WSGI application.
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Django WSGI application
from docpat.wsgi import application

# Export as 'app' for compatibility with 'gunicorn app:app'
app = application

