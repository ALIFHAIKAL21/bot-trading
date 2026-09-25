"""
Integration test for TradingView Chart Widget in Flowdev FRAME Workstation.
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from src.frame.gui.app import MainWindow
from src.frame.gui.worker import BacktestWorker

def run_integration_test():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()

    print("[TEST] MainWindow created and shown.")
    tv = win.tradingview_widget
    assert tv is not None, "TradingView widget must not be None"

    # Run quick worker test on 2026
    print("[TEST] Launching BacktestWorker for 2026...")
    worker = BacktestWorker(mode="2026", capital=500.0, lot=0.01)
    results = {}

    def on_finished(data):
        results['data'] = data
        print(f"[TEST] Finished signal received. Total trades: {data.get('total_trades')}")
        # Test loading data into GUI
        win._on_worker_finished(data)

        # Verify months populated
        month_count = tv.combo_month.count()
        print(f"[TEST] Months populated in TradingView combo: {month_count} months")
        assert month_count > 0, "TradingView months combo should have items"

        # Verify trades populated
        trade_count = tv.combo_trade.count()
        print(f"[TEST] Trades populated in TradingView combo: {trade_count} items")
        assert trade_count > 0, "TradingView trades combo should have items"

        # Test HUD and Selection
        if trade_count > 1:
            tv.combo_trade.setCurrentIndex(1)
            print(f"[TEST] Selected trade #1: {tv.lbl_hud_signal.text()[:40]}...")
            assert "SIGNAL" in tv.lbl_hud_signal.text()
            assert "PRICING" in tv.lbl_hud_pricing.text()
            assert "OMS" in tv.lbl_hud_oms.text()
            assert "REALIZED" in tv.lbl_hud_outcome.text()

        # Test Prev / Next Navigation
        initial_idx = tv.combo_trade.currentIndex()
        tv._on_next_trade()
        print(f"[TEST] Next trade index: {tv.combo_trade.currentIndex()}")
        assert tv.combo_trade.currentIndex() == initial_idx + 1

        tv._on_prev_trade()
        print(f"[TEST] Prev trade index: {tv.combo_trade.currentIndex()}")
        assert tv.combo_trade.currentIndex() == initial_idx

        # Test Filter switching (e.g. Wins Only)
        tv.combo_filter.setCurrentIndex(1) # Wins Only
        wins_count = len(tv.filtered_trades)
        print(f"[TEST] Filter Wins Only: {wins_count} trades")
        assert wins_count > 0, "Wins filter should find trades"

        # Test Double-click trade row jump from Journal tab
        win._on_trade_row_double_clicked(0, 0)
        assert win.tabs.currentIndex() == 1, "Double clicking a trade must switch to tab index 1 (TradingView)"
        print("[TEST] Double click trade jump passed: Switched to TradingView tab")

        # Test Snapshot Export
        tv._take_snapshot()
        snap_files = list(pathlib.Path(r"c:\Ngoding\bot_trading\reports\img").glob("tv_snapshot_*.png"))
        assert len(snap_files) > 0, "Snapshot PNG should be created in reports/img"
        print(f"[TEST] Snapshot export passed: {snap_files[-1].name}")

        # Test report export
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information = lambda parent, title, text: None
        win._export_reports()
        print("[TEST] Export reports executed successfully!")

        print("[TEST] ALL UPGRADED TRADINGVIEW INTEGRATION TESTS PASSED!")
        QTimer.singleShot(500, app.quit)

    def on_error(err):
        print(f"[TEST ERROR] {err}")
        app.quit()

    worker.finished_backtest.connect(on_finished)
    worker.error_occurred.connect(on_error)
    worker.start()

    app.exec()

if __name__ == "__main__":
    run_integration_test()
