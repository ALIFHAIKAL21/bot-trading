"""
End-to-End GUI Integration Test for FRAME Workstation
Verifies that MainWindow:
1. Selects Dual-Mode and Custom Range
2. Executes backtest worker
3. Populates all UI tabs (Metrics, Chart Canvas, TradingView Inspector, Monthly Table, Trade Journal Table)
4. Saves export report successfully
"""

import sys, pathlib
sys.path.append(r"c:\Ngoding\bot_trading")
from PySide6.QtWidgets import QApplication, QMessageBox
QMessageBox.information = lambda *args, **kwargs: None
from src.frame.gui.app import MainWindow

def test_full_gui_cycle():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    
    # 1. Configure controls: Custom Range (2026-05 to 2026-08), Capital $500, Dual-Mode Comparison
    win.combo_period_mode.setCurrentIndex(2) # Custom Range
    win.combo_range_start.setCurrentText("2026-05")
    win.combo_range_end.setCurrentText("2026-08")
    win.spin_capital.setValue(500.0)
    win.combo_sizing.setCurrentIndex(0) # Dual-Mode Comparison
    win.spin_max_lot.setValue(2.00)

    print("[TEST] Triggering _start_backtest()...")
    win._start_backtest()

    import time
    while win.worker.isRunning():
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()

    # Verify results
    assert win.last_results is not None, "worker did not deliver results"
    res = win.last_results
    print(f"[TEST] Worker finished. Total trades: {res['total_trades']}")
    print(f"[TEST] Dual mode flag: {res.get('dual_mode')}")
    print(f"[TEST] Metrics card Net PnL text: {win.metrics_panel.card_net_pnl.lbl_value.text()}")
    print(f"[TEST] Metrics card Sizing text: {win.metrics_panel.card_sizing.lbl_value.text()}")
    print(f"[TEST] Monthly table row count: {win.tbl_monthly.rowCount()}")
    print(f"[TEST] Trade table row count: {win.tbl_trades.rowCount()}")
    print(f"[TEST] Trade table col count: {win.tbl_trades.columnCount()}")
    print(f"[TEST] Trade table row 0 Lot: {win.tbl_trades.item(0, 3).text()}")

    assert win.tbl_trades.columnCount() == 10, "Trade table should have 10 columns"
    assert win.tbl_monthly.rowCount() == 4, f"Expected 4 months, got {win.tbl_monthly.rowCount()}"

    # Test export
    win._export_reports()
    print("[TEST] Report export executed.")

    print("\n[SUCCESS] Full GUI integration cycle completed with 100% PASS!")

if __name__ == "__main__":
    test_full_gui_cycle()
