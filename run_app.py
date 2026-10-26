"""
run_app.py - Launch the ML-NIDS Web Console locally using Uvicorn.
Usage: python run_app.py
"""

import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 65)
    print("STARTING ML-NIDS & EXPLAINABLE AI SOC DASHBOARD")
    print("=" * 65)
    print("URL: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press CTRL+C in this terminal to stop the server.")
    print("=" * 65)
    
    # Optionally trigger default browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000, reload=False)
