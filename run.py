import os
import subprocess
import sys

# Get port from environment variable for Render compatibility
port = int(os.environ.get("PORT", 8501))

# Build the command to run streamlit with the correct port
cmd = [
    "streamlit", "run", 
    "load_model.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--browser.serverAddress", "0.0.0.0",
    "--browser.gatherUsageStats", "false"
]

# Execute the command
process = subprocess.Popen(cmd)
process.wait()
