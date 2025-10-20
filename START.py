"""
🚀 HUIT Big Data Project - Super Simple Starter
Chỉ cần chạy: python START.py
"""

import subprocess
import sys
import os

def main():
    print("🚀 HUIT Big Data Project - Quick Start")
    print("=====================================")
    print()
    print("🌐 Starting web application...")
    print("📊 Dashboard: http://localhost:5000")
    print()
    print("💡 Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "simple_app.py"])
    except KeyboardInterrupt:
        print("\n✅ Application stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try running: pip install -r requirements.txt")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()