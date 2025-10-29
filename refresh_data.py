#!/usr/bin/env python3
"""
Data refresh script for Options Finder.
Updates options data from API sources.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

def refresh_options_data():
    """Refresh options data from API sources."""
    print("🔄 Starting data refresh...")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Check if ingest script exists
        ingest_script = Path("note/ingest_options.py")
        if not ingest_script.exists():
            print("❌ Data ingestion script not found: note/ingest_options.py")
            return False
        
        # Run the data ingestion script
        print("📡 Fetching fresh data from API...")
        result = subprocess.run([
            sys.executable, str(ingest_script)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data refreshed successfully!")
            
            # Clean up expired data
            print("🗑️ Cleaning up expired contracts...")
            cleanup_expired_data()
            
            print("📊 New data is now available in the app")
            return True
        else:
            print(f"❌ Error refreshing data:")
            print(f"   {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during data refresh: {e}")
        return False


def cleanup_expired_data():
    """Remove expired options from the data file."""
    try:
        import pandas as pd
        from datetime import date
        
        data_file = Path("data/latest/options_latest.parquet")
        if not data_file.exists():
            print("❌ No data file found to clean")
            return
        
        # Load the data
        df = pd.read_parquet(data_file)
        original_count = len(df)
        
        # Get current date
        current_date = date.today()
        
        # Filter out expired options
        df['expiry_date_parsed'] = pd.to_datetime(df['expiry_date']).dt.date
        active_df = df[df['expiry_date_parsed'] >= current_date].copy()
        active_df = active_df.drop('expiry_date_parsed', axis=1)
        
        # Save cleaned data
        if len(active_df) < original_count:
            active_df.to_parquet(data_file, index=False)
            removed_count = original_count - len(active_df)
            print(f"🗑️ Removed {removed_count} expired contracts")
        else:
            print("✅ No expired contracts found")
            
    except Exception as e:
        print(f"❌ Error cleaning expired data: {e}")

def check_data_age():
    """Check how old the current data is."""
    data_file = Path("data/latest/options_latest.parquet")
    
    if not data_file.exists():
        print("❌ No data file found")
        return None
    
    try:
        file_time = data_file.stat().st_mtime
        current_time = datetime.now().timestamp()
        age_hours = (current_time - file_time) / 3600
        
        print(f"📅 Data file age: {age_hours:.1f} hours")
        
        if age_hours < 1:
            print("✅ Data is fresh")
        elif age_hours < 24:
            print("⚠️ Data is getting stale")
        else:
            print("❌ Data is very old - refresh recommended")
        
        return age_hours
        
    except Exception as e:
        print(f"❌ Error checking data age: {e}")
        return None

def main():
    """Main function."""
    print("🚀 Options Finder - Data Refresh Tool")
    print("=" * 50)
    
    # Check current data age
    age = check_data_age()
    
    # Ask user if they want to refresh
    if age is not None and age < 1:
        print("\n🤔 Data is already fresh. Refresh anyway? (y/n): ", end="")
        response = input().lower().strip()
        if response not in ['y', 'yes']:
            print("👋 No refresh needed. Exiting.")
            return
    
    # Refresh the data
    print("\n" + "=" * 50)
    success = refresh_options_data()
    
    if success:
        print("\n🎉 Data refresh completed successfully!")
        print("🌐 You can now refresh your browser to see the latest data")
    else:
        print("\n💥 Data refresh failed. Check the errors above.")

if __name__ == "__main__":
    main()
