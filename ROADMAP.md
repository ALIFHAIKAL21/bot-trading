# Quantitative Trading System Roadmap (ROADMAP.md)

This roadmap outlines strategic research directions, alpha modeling expansions, and execution infrastructure upgrades designed to transition the quantitative engine from single-asset trend following to a diversified multi-asset statistical arbitrage and market-neutral platform.

---

## 1. Multi-Asset Universe & Cross-Sectional Alpha
- **Objective**: Expand from single-asset `BTC/USDT` to a diversified liquid crypto universe (`BTC`, `ETH`, `SOL`, `BNB`, `AVAX`, `NEAR`).
- **Quant Rationale**: Single-asset strategies are vulnerable to prolonged non-trending or regime-locked conditions (such as the locked test period where directional edge was insufficient to beat friction). Cross-sectional strategies harvest relative value (long strongest relative momentum, short weakest) with market-neutral beta.
- **Milestones**:
  - [ ] Dynamic liquidity universe selector: rolling 30-day median dollar volume filter ($> \$50\text{M}$/day).
  - [ ] Cross-sectional feature pipeline: rolling rank transform, cross-sectional z-score of momentum, realized volatility, and basis.
  - [ ] Pair trading and cointegration engine (Engle-Granger, Johansen test) with Ornstein-Uhlenbeck mean-reversion modeling.

---

## 2. Perpetual Futures Funding Rate & Basis Arbitrage
- **Objective**: Harvest structural risk premia from perpetual swap funding rates and quarterly futures basis.
- **Quant Rationale**: Perpetual funding rates represent an asymmetric, positive-expectation yield paid by leveraged retail speculators to liquidity providers. In crypto bull markets, annualized funding rates frequently range between 15% and 40%.
- **Milestones**:
  - [ ] Funding rate ingestion pipeline capturing 8-hour historical rates across exchanges (Binance, Bybit, OKX).
  - [ ] Cash-and-carry basis strategy: long spot, short perpetual when annualized funding rate exceeds transaction cost threshold ($> 15\%$).
  - [ ] Negative funding rate reversal capture during market panics.

---

## 3. Microstructure & High-Frequency Order Book Features
- **Objective**: Incorporate Level 2 (L2) and Level 3 (L3) order book dynamics to enhance short-horizon prediction and reduce execution slippage.
- **Quant Rationale**: 1-hour OHLCV candles compress critical order book imbalances, aggressive buyer/seller ratios, and order cancellations that precede price inflections.
- **Milestones**:
  - [ ] Depth of Market (DOM) feature extraction: Order Book Imbalance (OBI), micro-price, bid-ask spread elasticity.
  - [ ] Trade tick flow analysis: Volume Synchronized Probability of Toxicity (VPIN) and Kyle's lambda.
  - [ ] Order flow toxicity risk veto: pause execution when adverse selection risk exceeds acceptable percentiles.

---

## 4. Execution Algorithms & Dynamic Frictions
- **Objective**: Replace discrete market orders with intelligent execution algorithms (TWAP, VWAP, POV) and limit-order queue placement.
- **Quant Rationale**: 10 bps taker fees + 5 bps slippage create severe cost drag that overwhelmed subtle 1-hour alpha signals during the out-of-sample test. Crossing the spread as a maker (-1 to 2 bps fee) fundamentally alters strategy viability.
- **Milestones**:
  - [ ] Passive limit order execution with queue-position tracking and cancellation logic.
  - [ ] Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP) execution slicing.
  - [ ] Reinforcement Learning (RL) execution agent trained in gym environment with realistic limit-order fill simulators.

---

## 5. Alternative Data & NLP Sentiment Ingestion
- **Objective**: Expand Model D beyond standard RSS feeds into low-latency event-driven signals.
- **Quant Rationale**: Public RSS feeds suffer from 5-15 minute reporting latency, rendering them degraded for proactive trade entry.
- **Milestones**:
  - [ ] Social volume and sentiment spike detection (LunarCrush, CryptoQuant on-chain metrics).
  - [ ] On-chain whale transfer and exchange inflow/outflow monitoring.
  - [ ] Macro event calendar scraper with automatic pre-FOMC/CPI position de-risking.
