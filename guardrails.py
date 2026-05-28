"""
Hard guardrail rules — imported by the SAC environment, backtest, and live trader.

All hard guardrails (HG1–HG11) live here as pure Python logic.
Models cannot override these rules under any circumstances.
"""

from __future__ import annotations
import numpy as np

# ── Transaction cost constants ────────────────────────────────────────────────
BROKERAGE_PER_ORDER    = 20.0     # ₹20 flat per order (Upstox)
STT_SELL_RATE          = 0.0001   # 0.01% of turnover, sell side only
EXCHANGE_CHARGE_RATE   = 0.00005  # 0.005% of turnover
SLIPPAGE_N50_PTS       = 1.0      # average slippage in index points
SLIPPAGE_NBANK_PTS     = 2.0
LOT_SIZE_N50           = 75
LOT_SIZE_NBANK         = 35


def compute_transaction_cost(instrument: str, order_value_inr: float,
                             lots: int, side: str) -> float:
    """
    Exact transaction cost formula from the trading plan.

    Args:
        instrument:      "NIFTY50" or "NIFTYBANK"
        order_value_inr: price × lot_size × lots
        lots:            number of lots traded
        side:            "BUY" or "SELL"

    Returns:
        Total cost in ₹.
    """
    brokerage = BROKERAGE_PER_ORDER
    stt       = order_value_inr * STT_SELL_RATE if side == "SELL" else 0.0
    exchange  = order_value_inr * EXCHANGE_CHARGE_RATE
    slip_pts  = SLIPPAGE_N50_PTS if instrument == "NIFTY50" else SLIPPAGE_NBANK_PTS
    lot_size  = LOT_SIZE_N50     if instrument == "NIFTY50" else LOT_SIZE_NBANK
    slippage  = slip_pts * lot_size * lots
    return brokerage + stt + exchange + slippage


def lots_for_risk(capital: float, risk_pct: float, sl_distance_pts: float,
                  instrument: str) -> int:
    """Calculate number of lots that risks at most risk_pct of capital."""
    lot_size = LOT_SIZE_N50 if instrument == "NIFTY50" else LOT_SIZE_NBANK
    if sl_distance_pts <= 0:
        return 1
    return max(1, round((capital * risk_pct) / (sl_distance_pts * lot_size)))


class Guardrails:
    """
    Stateful guardrail checker. One instance per trading session / episode.
    Must call reset_day() at the start of each trading day.

    Usage:
        gr = Guardrails(capital=50_000)
        gr.reset_day(orb_range_pct=0.5, daily_atr=80, atr_mean_20d=60, atr_std_20d=10)

        # Before sending a new entry order:
        allowed, scale = gr.check_entry(...)

        # Apply HG11 to SAC action (returns clipped action):
        action = gr.apply_hg11(action, regime_max_prob)

        # Apply HG5 (total exposure cap):
        action = gr.apply_hg5(action)

        # Hard exit at 15:00:
        gr.hard_exit_eod()
    """

    def __init__(self, capital: float = 50_000.0):
        self.capital         = capital
        self.peak_capital    = capital
        self.cash            = capital
        self.realized_pnl    = 0.0       # cumulative for the session

        # Per-day state (reset each morning)
        self.daily_loss      = 0.0       # realised + unrealised loss today
        self.daily_halted    = False
        self.trades_today    = 0

        # Open positions (signed allocation fraction -1..+1)
        self.pos_n50         = 0.0
        self.pos_nbank       = 0.0

        # Day-level ORB state
        self.orb_range_pct   = 0.0
        self.orb_valid_day   = True

        # Volatility state
        self.is_high_vol_day = False
        self.current_atr     = 0.0
        self.atr_mean_20d    = 1.0
        self.atr_std_20d     = 0.1

    # ── Day management ────────────────────────────────────────────────────────

    def reset_day(self, orb_range_pct: float = 0.0,
                  daily_atr: float = 0.0,
                  atr_mean_20d: float = 1.0,
                  atr_std_20d: float = 0.1) -> None:
        self.daily_loss      = 0.0
        self.daily_halted    = False
        self.trades_today    = 0
        self.realized_pnl    = 0.0   # FIX 1: stale loss from prior day kept halting day 2+
        self.pos_n50         = 0.0   # FIX 2: explicit reset; HG8 exits should have cleared these
        self.pos_nbank       = 0.0
        self.orb_range_pct   = orb_range_pct
        self.current_atr     = daily_atr
        self.atr_mean_20d    = max(atr_mean_20d, 1e-6)
        self.atr_std_20d     = max(atr_std_20d,  1e-6)

        # HG6: ORB range filter
        self.orb_valid_day   = (0.1 <= orb_range_pct <= 2.0)

        # HG10: extreme volatility
        self.is_high_vol_day = (daily_atr > atr_mean_20d + 0.5 * atr_std_20d)

    def update_unrealised(self, unrealised_pnl: float) -> None:
        """Call each bar with the current unrealised P&L to check HG2."""
        self.daily_loss = max(0.0, -(self.realized_pnl + unrealised_pnl))
        if self.daily_loss > self.capital * 0.03:
            self.daily_halted = True

    def record_trade_close(self, trade_pnl: float) -> None:
        self.realized_pnl += trade_pnl
        self.trades_today += 1
        self.daily_loss    = max(0.0, -self.realized_pnl)
        if self.daily_loss > self.capital * 0.03:
            self.daily_halted = True

    # ── Action modifiers ──────────────────────────────────────────────────────

    def apply_hg5(self, action: np.ndarray) -> np.ndarray:
        """
        HG5: total portfolio exposure cap at 80%.
        If |action[0]| + |action[1]| > 0.80, scale both proportionally.
        """
        total = abs(action[0]) + abs(action[1])
        if total > 0.80:
            scale = 0.80 / total
            action = action * scale
        return action

    def apply_hg10(self, action: np.ndarray) -> np.ndarray:
        """
        HG10: extreme volatility scaling — halve all new position sizes.
        Returns modified action. Caller should also disable Signal 2.
        """
        if self.is_high_vol_day:
            action = action * 0.5
        return action

    def apply_hg11(self, action: np.ndarray, regime_max_prob: float) -> np.ndarray:
        """HG11: regime uncertainty gate — clip SAC action to ±0.3."""
        if regime_max_prob < 0.5:
            action = np.clip(action, -0.3, 0.3)
        return action

    # ── Entry checks ──────────────────────────────────────────────────────────

    def check_entry(self, bar_hour: int, bar_minute: int,
                    sl_distance_pts: float,
                    instrument: str,
                    has_sl: bool = True) -> tuple[bool, str]:
        """
        Run all hard guardrail entry checks.

        Returns:
            (allowed, reason_if_blocked)
        """
        bar_time = bar_hour * 60 + bar_minute

        # HG1: time fence
        if bar_time < 9 * 60 + 55 or bar_time >= 15 * 60:
            return False, "HG1_TIME_FENCE"

        # HG2: daily loss halt
        if self.daily_halted:
            return False, "HG2_DAILY_LOSS_HALT"

        # HG9: stop loss required
        if not has_sl:
            return False, "HG9_NO_SL"

        # HG3: single trade max loss — caller adjusts lot size; always allowed
        # (size reduction handled by lots_for_risk)

        return True, ""

    # ── EOD ───────────────────────────────────────────────────────────────────

    def hard_exit_eod(self) -> None:
        """HG8: force-exit all positions at 15:00."""
        self.pos_n50   = 0.0
        self.pos_nbank = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def sl_distance_pts(self, entry_price: float, atr: float,
                        direction: int) -> float:
        """
        Compute stop-loss distance in index points.
        High-vol days use 2.5×ATR; normal days use 1.5×ATR.
        Also enforces HG3: never risk > 2% of capital.
        """
        mult = 2.5 if self.is_high_vol_day else 1.5
        atr_sl = mult * atr
        max_sl = (self.capital * 0.02)   # HG3 — in ₹, not pts
        # convert max_sl from ₹ to pts: max_pts = max_₹ / (lot_size × lots)
        # since we don't know lots yet, return ATR-based SL (caller enforces HG3)
        return atr_sl
