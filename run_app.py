#!/usr/bin/env python3
"""
Startup script for Options Finder application.
Provides easy commands to run different components.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_streamlit():
    """Run the Streamlit application."""
    print("🚀 Starting Options Finder Streamlit App...")
    print("📊 Open your browser to: http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])


def run_api():
    """Run the FastAPI server."""
    print("🔌 Starting Options Finder API Server...")
    print("📡 API available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    subprocess.run([sys.executable, "api/server.py"])


def run_tests():
    """Run the test suite."""
    print("🧪 Running Options Finder Test Suite...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def check_data():
    """Check if data files exist."""
    data_path = Path("data/latest/options_latest.parquet")
    if data_path.exists():
        print(f"✅ Data file found: {data_path}")
        return True
    else:
        print(f"❌ Data file not found: {data_path}")
        print("📁 Available data files:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.rglob("*.parquet"):
                print(f"   - {file}")
        return False


def install_deps():
    """Install dependencies."""
    print("📦 Installing Options Finder dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def main():
    """Main function with command line interface."""
    if len(sys.argv) < 2:
        print("Options Finder - Full Stack Options Analysis Platform")
        print("\nUsage:")
        print("  python run_app.py streamlit    - Run Streamlit UI")
        print("  python run_app.py api          - Run FastAPI server")
        print("  python run_app.py test         - Run test suite")
        print("  python run_app.py check        - Check data files")
        print("  python run_app.py refresh      - Refresh data from API")
        print("  python run_app.py cleanup      - Clean up expired contracts")
        print("  python run_app.py train        - Train ML model")
        print("  python run_app.py install      - Install dependencies")
        print("  python run_app.py all          - Run both UI and API")
        return
    
    command = sys.argv[1].lower()
    
    if command == "streamlit":
        run_streamlit()
    elif command == "api":
        run_api()
    elif command == "test":
        run_tests()
    elif command == "check":
        check_data()
    elif command == "install":
        install_deps()
    elif command == "refresh":
        print("🔄 Refreshing options data...")
        subprocess.run([sys.executable, "refresh_data.py"])
    elif command == "cleanup":
        print("🗑️ Cleaning up expired contracts...")
        subprocess.run([sys.executable, "refresh_data.py", "cleanup"])
    elif command == "train":
        print("🤖 Training ML model...")
        subprocess.run([sys.executable, "train_ml_model.py"])
    elif command == "all":
        print("🚀 Starting both Streamlit UI and FastAPI server...")
        print("📊 Streamlit UI: http://localhost:8501")
        print("📡 API Server: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both servers")
        
        # Run both in parallel (simplified approach)
        import threading
        import time
        
        def run_streamlit_thread():
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
        
        def run_api_thread():
            subprocess.run([sys.executable, "api/server.py"])
        
        # Start both servers
        streamlit_thread = threading.Thread(target=run_streamlit_thread)
        api_thread = threading.Thread(target=run_api_thread)
        
        streamlit_thread.start()
        time.sleep(2)  # Give Streamlit time to start
        api_thread.start()
        
        try:
            streamlit_thread.join()
            api_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python run_app.py' for usage information")


if __name__ == "__main__":
    main()
