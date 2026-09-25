"""
Test GUI simulation flow:
1. Single Month (Dual-Mode Comparison)
2. Custom Multi-Month Range (Dynamic Compounding)
3. Annual with custom capital ($250) (Flat 0.01)
"""

import sys, pathlib, time
sys.path.append(r"c:\Ngoding\bot_trading")
from PySide6.QtWidgets import QApplication
from src.frame.gui.worker import BacktestWorker

def test_single_month_dual():
    print("\n--- Test 1: Single Month (2026-08) Dual-Mode ---")
    w = BacktestWorker(
        mode="Month 2026-08",
        start_date="2026-08",
        end_date="2026-08",
        capital=500.0,
        lot=0.01,
        sizing_mode="dual",
        max_lot=2.0
    )
    result = {}
    def on_finished(res):
        result.update(res)
    w.finished_backtest.connect(on_finished)
    w.run()
    
    assert 'trades' in result, "No trades in result"
    print(f"Total trades in 2026-08: {result['total_trades']}")
    print(f"Flat PnL: ${result.get('net_pnl_flat'):+,.2f} | Dyn PnL: ${result.get('net_pnl_dyn'):+,.2f}")
    print(f"Peak Lot: {result['peak_lot']:.2f}L")
    print("Dual curves present:", result.get('equity_curve_flat') is not None, result.get('equity_curve_dyn') is not None)
    print("Test 1 PASS!")

def test_custom_range_dynamic():
    print("\n--- Test 2: Custom Range (2026-01 to 2026-04) Dynamic Compounding ($1,000) ---")
    w = BacktestWorker(
        mode="2026-01 to 2026-04",
        start_date="2026-01",
        end_date="2026-04",
        capital=1000.0,
        lot=0.01,
        sizing_mode="dynamic",
        max_lot=3.0
    )
    result = {}
    def on_finished(res):
        result.update(res)
    w.finished_backtest.connect(on_finished)
    w.run()
    
    assert 'trades' in result, "No trades in result"
    print(f"Total trades: {result['total_trades']}")
    print(f"Net PnL: ${result['net_pnl']:+,.2f} ({result['return_pct']:+,.1f}%)")
    print(f"Max DD: {result['max_drawdown']:.2f}% | Peak Lot: {result['peak_lot']:.2f}L")
    print(f"Monthly breakdown count: {len(result['monthly'])}")
    for m in result['monthly']:
        print(f"  Month: {m['month']} | Trades: {m['trades']} | WR: {m['win_rate']}% | PnL: ${m['pnl']:+,.2f}")
    print("Test 2 PASS!")

def test_flat_stress_capital():
    print("\n--- Test 3: 2026 Flat 0.01 with $250 Capital ---")
    w = BacktestWorker(
        mode="2026",
        start_date="",
        end_date="",
        capital=250.0,
        lot=0.01,
        sizing_mode="flat",
        max_lot=1.0
    )
    result = {}
    def on_finished(res):
        result.update(res)
    w.finished_backtest.connect(on_finished)
    w.run()
    
    assert 'trades' in result, "No trades in result"
    print(f"Total trades: {result['total_trades']}")
    print(f"Net PnL: ${result['net_pnl']:+,.2f} ({result['return_pct']:+,.1f}%)")
    print(f"Final Balance: ${result['final_equity']:,.2f}")
    print("Test 3 PASS!")

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    test_single_month_dual()
    test_custom_range_dynamic()
    test_flat_stress_capital()
    print("\nALL SIMULATION TESTS PASSED SUCCESSFULLY!")
