"""
Test Script for FLOWDEV FRAME Live Paper Trading Engine
Verifies:
1. Paper Broker (order entry, floating PnL, kinetic ratcheting, SL/TP execution)
2. Live Market Feed (synthetic/fallback ticks, candle updates)
3. Live Agent (LoRA model inference, OMS stage state machine)
4. Full Integration without launching blocking GUI
"""

import sys, pathlib, time

current_dir = pathlib.Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.frame.live.paper_broker import LivePaperBroker
from src.frame.live.feed import LiveMarketFeed
from src.frame.live.agent import LiveAgent

def test_paper_broker():
    print("\n--- TEST 1: Paper Broker Order Lifecycle & Kinetic OMS ---")
    broker = LivePaperBroker(initial_capital=500.0, lot_mode="dynamic", max_lot=2.0)
    assert broker.cash == 500.0
    assert broker.open_position is None

    # Open BUY trade (sl_dist = 15.0)
    pos = broker.open_order(
        direction="BUY",
        current_bid=4300.0,
        current_ask=4300.3,
        sl_dist=15.0,
        reason="TestSignal"
    )
    assert pos is not None
    assert broker.open_position is not None
    assert pos["lot"] >= 0.01
    print(f"  [OK] Opened BUY: Lots={pos['lot']:.2f}, Entry={pos['entry_price']}, SL={pos['current_sl']}, TP={pos['current_tp']}")

    # Agent OMS Controller test
    agent = LiveAgent(broker=broker)

    # Tick update in profit (+0.8R -> triggers Stage 1 Micro-BE)
    tick = {"bid": 4312.5, "ask": 4312.8, "last": 4312.5, "spread": 0.3, "time": time.time()}
    agent.on_tick(tick)
    print(f"  [OK] After +0.8R: Floating PnL=${pos['floating_pnl']:.2f}, R={pos['floating_r']:.2f}R, BE={pos['be_activated']}, SL={pos['current_sl']}")
    assert pos["be_activated"] is True
    assert pos["current_sl"] >= pos["entry_price"]

    # Tick update in higher profit (+1.3R -> triggers Stage 2 Smart Ratchet)
    tick = {"bid": 4320.5, "ask": 4320.8, "last": 4320.5, "spread": 0.3, "time": time.time()}
    agent.on_tick(tick)
    print(f"  [OK] After +1.3R: Floating PnL=${pos['floating_pnl']:.2f}, R={pos['floating_r']:.2f}R, Ratchet={pos['ratchet_activated']}, SL={pos['current_sl']}")
    assert pos["ratchet_activated"] is True
    assert pos["current_sl"] >= pos["entry_price"] + (0.5 * 15.0)

    # Tick update hitting TP (+2.7R)
    tp_price = pos["current_tp"]
    tick = {"bid": tp_price + 0.5, "ask": tp_price + 0.8, "last": tp_price + 0.5, "spread": 0.3, "time": time.time()}
    agent.on_tick(tick)
    assert broker.open_position is None
    assert len(broker.trade_history) == 1
    closed = broker.trade_history[0]
    print(f"  [OK] Closed trade: Exit={closed['exit_price']}, Realized PnL=${closed['net_pnl']:.2f}, Balance=${broker.cash:.2f}, Reason={closed['exit_reason']}")
    assert broker.cash > 500.0

def test_market_feed():
    print("\n--- TEST 2: Live Market Feed & Tick Generation ---")
    try:
        from PySide6.QtCore import QCoreApplication
    except ImportError:
        from PyQt6.QtCore import QCoreApplication
        
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    
    feed = LiveMarketFeed(symbol="XAUUSD", poll_interval_ms=30)
    ticks_received = []

    def on_tick(tick):
        ticks_received.append(tick)

    feed.tick_received.connect(on_tick)
    feed.set_source("emulator") # use emulator for deterministic test
    feed.start()

    t_end = time.time() + 0.35
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)

    feed.stop()

    assert len(ticks_received) > 0
    first_tick = ticks_received[0]
    print(f"  [OK] Captured {len(ticks_received)} live ticks. Sample: Bid={first_tick['bid']:.2f}, Ask={first_tick['ask']:.2f}, Spread={first_tick['spread']:.2f}")

def test_live_agent_model():
    print("\n--- TEST 3: Live Agent LoRA Neural Inference ---")
    broker = LivePaperBroker(initial_capital=500.0)
    agent = LiveAgent(broker=broker)

    print(f"  [OK] Agent Device: {agent.device}, Model Loaded: {agent.model is not None}")
    
    # Generate 65 simulated M30 candles
    import numpy as np
    
    np.random.seed(42)
    prices = 4300.0 + np.cumsum(np.random.randn(65) * 2.0)
    candles = []
    base_t = int(time.time()) - (65 * 1800)
    for i in range(65):
        p = float(prices[i])
        candles.append({
            "time": base_t + (i * 1800),
            "open": p,
            "high": p + 2.0,
            "low": p - 2.0,
            "close": p + 0.5,
            "volume": 500
        })
    
    # Run inference
    start_t = time.perf_counter()
    action, conf, probs = agent._infer_signal(candles)
    inf_ms = (time.perf_counter() - start_t) * 1000.0
    print(f"  [OK] Neural Inference: Action={action}, Conf={conf*100:.1f}%, Latency={inf_ms:.2f}ms")
    assert action in ["BUY", "SELL", "HOLD"]
    assert conf > 0.0

if __name__ == "__main__":
    test_paper_broker()
    test_market_feed()
    test_live_agent_model()
    print("\n=======================================================")
    print(" >>> ALL LIVE TRADING ENGINE TESTS PASSED PERFECTLY! <<< ")
    print("=======================================================\n")
