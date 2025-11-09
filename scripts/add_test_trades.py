"""
Script to add test/demo trade data to the database
This is useful for testing the dashboard without running live trades
"""
import sys
import os
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import init_db, SessionLocal, Trade

def add_test_trades():
    """Add realistic test trades to the database"""
    
    # Initialize database
    init_db()
    db = SessionLocal()
    
    # Check if trades already exist
    existing_trades = db.query(Trade).count()
    if existing_trades > 0:
        print(f"⚠️  Database already has {existing_trades} trades.")
        response = input("Do you want to add more test trades? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            db.close()
            return
    
    # Generate test trades
    print("\n🔨 Generating test trades...")
    
    base_time = datetime.now() - timedelta(days=30)  # Start 30 days ago
    base_price = 43000  # Starting BTC price
    
    trades = []
    position = None  # Track if we have an open position
    
    for i in range(50):  # Generate 50 test trades
        # Simulate price movement
        price_change = random.uniform(-0.05, 0.05)  # -5% to +5%
        current_price = base_price * (1 + price_change)
        
        # Determine trade side
        if position is None:
            # No position, place a BUY
            side = 'BUY'
            position = current_price
        else:
            # Have position, decide whether to SELL
            if random.random() > 0.3:  # 70% chance to sell
                side = 'SELL'
                position = None
            else:
                # Hold position, no trade
                base_price = current_price
                base_time += timedelta(hours=random.randint(1, 24))
                continue
        
        trade = Trade(
            symbol='BTCUSDT',
            side=side,
            quantity=0.001 + random.uniform(-0.0002, 0.0002),  # 0.0008 to 0.0012 BTC
            price=current_price,
            timestamp=base_time + timedelta(hours=i * random.randint(1, 12)),
            order_id=f"TEST_{i:04d}_{random.randint(1000, 9999)}"
        )
        
        trades.append(trade)
        base_price = current_price
    
    # Add all trades to database
    try:
        db.add_all(trades)
        db.commit()
        print(f"✅ Successfully added {len(trades)} test trades to the database!")
        
        # Show summary
        buy_trades = sum(1 for t in trades if t.side == 'BUY')
        sell_trades = sum(1 for t in trades if t.side == 'SELL')
        
        print(f"\n📊 Summary:")
        print(f"   Total Trades: {len(trades)}")
        print(f"   BUY Trades:   {buy_trades}")
        print(f"   SELL Trades:  {sell_trades}")
        print(f"   Symbol:       BTCUSDT")
        print(f"   Date Range:   {trades[0].timestamp.strftime('%Y-%m-%d')} to {trades[-1].timestamp.strftime('%Y-%m-%d')}")
        
        # Calculate approximate P&L
        buy_value = sum(t.price * t.quantity for t in trades if t.side == 'BUY')
        sell_value = sum(t.price * t.quantity for t in trades if t.side == 'SELL')
        pnl = sell_value - buy_value
        
        print(f"\n💰 Approximate P&L: ${pnl:.2f}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error adding trades: {e}")
    finally:
        db.close()
    
    print("\n✅ Done! Refresh your Streamlit dashboard to see the trades.")
    print("   Dashboard URL: http://localhost:8501")

if __name__ == "__main__":
    print("="*50)
    print("🧪 Test Trade Data Generator")
    print("="*50)
    print("\nThis script will add realistic test trades to your database.")
    print("This is useful for testing the dashboard functionality.\n")
    
    add_test_trades()
