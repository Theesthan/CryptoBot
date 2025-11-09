"""
Script to clear all test trades from the database
WARNING: This will delete ALL trades in your database!
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import init_db, SessionLocal, Trade

def clear_all_trades():
    """Remove all trades from the database"""
    
    init_db()
    db = SessionLocal()
    
    try:
        # Count existing trades
        trade_count = db.query(Trade).count()
        
        if trade_count == 0:
            print("✅ Database is already empty. No trades to delete.")
            return
        
        print(f"\n⚠️  WARNING: You are about to delete {trade_count} trades from the database!")
        print("This action CANNOT be undone.")
        response = input("\nAre you sure you want to continue? (yes/no): ")
        
        if response.lower() != 'yes':
            print("❌ Cancelled. No trades were deleted.")
            return
        
        # Delete all trades
        deleted = db.query(Trade).delete()
        db.commit()
        
        print(f"\n✅ Successfully deleted {deleted} trades from the database!")
        print("\n💡 You can add new test trades by running:")
        print("   python scripts\\add_test_trades.py")
        print("   or")
        print("   add_test_trades.bat")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting trades: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("="*50)
    print("🗑️  Clear All Trades from Database")
    print("="*50)
    print("\nThis script will DELETE all trades in your database.")
    print("Use this to start fresh or remove test data.\n")
    
    clear_all_trades()
