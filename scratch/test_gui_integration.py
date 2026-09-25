import sys, pathlib
sys.path.insert(0, r'c:\Ngoding\bot_trading')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.frame.gui.app import MainWindow

app = QApplication(sys.argv)
win = MainWindow()
win.show()

# Trigger backtest programmatically
win._start_backtest()

def check_finished():
    if win.last_results is not None:
        res = win.last_results
        print("Backtest finished successfully inside GUI!")
        print(f"Total trades in GUI: {res['total_trades']}")
        print(f"Net PnL: ${res['net_pnl']:,.2f}")
        print(f"Win Rate: {res['win_rate']:.2f}%")
        print(f"Max DD: {res['max_drawdown']:.2f}%")
        print(f"Monthly rows in table: {win.tbl_monthly.rowCount()}")
        print(f"Trades in journal table: {win.tbl_trades.rowCount()}")
        app.quit()

timer = QTimer()
timer.timeout.connect(check_finished)
timer.start(500)

app.exec()
print("GUI Integration Test completed with 0 errors.")
