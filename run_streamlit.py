"""
Streamlit app runner script
Run this file to start the Streamlit dashboard
"""
import os
import sys

# Ensure the app runs from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run streamlit
os.system("streamlit run src/streamlit_app.py --server.port 8501 --server.address 0.0.0.0")
