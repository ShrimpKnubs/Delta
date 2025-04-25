# --- START OF feature_engineering.py (V4 - Explicit Heuristics) ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for the Full Moon Turning Point Detection System.
V4: Adds explicit features based on user heuristics (Volume/Price action combos,
short-term Z-score) on top of multi-scale features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from functools import lru_cache
import logging
from pathlib import Path

# Local imports
try:
    script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    src_dir = script_dir.parent / 'src'
    if not src_dir.is_dir():
        src_dir = script_dir # Fallback if src is not in parent
    import sys
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir)) # Add src directory itself
    if str(src_dir.parent) not in sys.path:
         sys.path.insert(0, str(src_dir.parent)) # Add project root

    from lunar_phase_calculator import LunarPhaseCalculator
except ImportError as e:
    print(f"Error importing LunarPhaseCalculator: {e}. Check path: {src_dir}")
    sys.exit(1)


# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """
    V4: Creates features including multi-scale indicators AND explicit heuristics.
    """

    def __init__(self, lunar_calculator: Optional[LunarPhaseCalculator] = None):
        if lunar_calculator is None:
            self.lunar_calculator = LunarPhaseCalculator()
        else:
            self.lunar_calculator = lunar_calculator
        logger.info("FeatureEngineer initialized (V4 - Explicit Heuristics).")
        self.epsilon = 1e-9
        # Define multi-scale periods
        self.periods_short = 7
        self.periods_medium = 14
        self.periods_long = 28
        self.curve_windows = [3, 7, 15]
        # Define thresholds/parameters for heuristic features
        self.short_z_window = 15 # Window for short-term volume Z-score
        self.small_body_threshold = 0.3 # Body size < 30% of medium ATR
        self.vol_exhaust_days = 3 # Consecutive lower volume days for exhaustion signal
        self.price_curve_threshold = 0.1 # Threshold for price curve signal


    def create_features(self, data: pd.DataFrame, include_target: bool = False,
                       known_turning_points: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Create V4 feature set including multi-scale and explicit heuristics.
        """
        features = data.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in features.columns]
        if missing_columns: raise ValueError(f"Missing required columns: {missing_columns}")

        logger.debug("Adding lunar features...")
        features = self._add_lunar_features(features)

        logger.debug("Adding multi-scale volume features...")
        features = self._add_volume_features(features)

        logger.debug("Adding multi-scale price action features...")
        features = self._add_price_action_features(features)

        logger.debug("Adding multi-scale technical indicators...")
        features = self._add_technical_indicators(features)

        logger.debug("Adding interaction and explicit heuristic features (V4)...")
        features = self._add_interaction_and_heuristic_features(features) # Combined step

        if include_target:
            if known_turning_points:
                logger.debug("Adding target variable...")
                features = self._add_target_variable(features, known_turning_points)
            else:
                logger.warning("`include_target=True` but `known_turning_points` not provided.")

        # Define columns to drop (ensure new heuristic base features are kept if needed elsewhere)
        cols_to_drop_final = [
             'open', 'high', 'low', 'close', 'volume',
             'lunar_phase_name', 'true_range',
             'volume_ema_5', 'volume_ema_10', 'volume_ema_20', 'volume_ema_30', 'volume_ema_50', 'volume_ema_100',
             'volume_higher', 'volume_lower'
             ]
        cols_to_drop_final.extend(['month', 'day_of_month', 'day_of_week', 'day_of_year']) # Remove date parts if they exist

        if include_target:
            target_cols = ['is_turning_point', 'is_top', 'is_bottom']
            cols_to_drop_final = [col for col in cols_to_drop_final if col not in target_cols]

        features_before_drop = set(features.columns)
        features = features.drop(columns=[col for col in cols_to_drop_final if col in features.columns], errors='ignore')
        features_after_drop = set(features.columns)
        dropped_cols_list = list(features_before_drop - features_after_drop)
        logger.debug(f"Columns dropped before final output: {dropped_cols_list}")

        initial_rows = len(features)
        features = features.dropna()
        dropped_rows = initial_rows - len(features)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows containing NaNs after feature calculation.")

        logger.info(f"Explicit Heuristic Feature generation V4 complete. Shape: {features.shape}")
        return features

    # --- Lunar, Volume, Price Action, Technical Indicators (Keep V3 versions) ---
    def _add_lunar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # --- Keep this method exactly the same as V3 ---
        df = data.copy()
        lunar_info_series = df.index.to_series().apply(self.lunar_calculator.calculate_lunar_phase)
        df['lunar_phase_pct'] = lunar_info_series.apply(lambda x: x['phase_percentage'])
        df['lunar_phase_name'] = lunar_info_series.apply(lambda x: x['phase_name'])
        df['days_to_full_moon'] = lunar_info_series.apply(lambda x: x['days_to_full_moon'])
        df['is_near_full_moon'] = lunar_info_series.apply(lambda x: x['is_near_full_moon']).astype(int)
        df['lunar_phase_sin'] = np.sin(2 * np.pi * df['lunar_phase_pct'] / 100)
        df['lunar_phase_cos'] = np.cos(2 * np.pi * df['lunar_phase_pct'] / 100)
        df['full_moon_day'] = (abs(df['days_to_full_moon']) < 1).astype(int)
        df['full_moon_window_3d'] = (abs(df['days_to_full_moon']) <= 3).astype(int)
        df['abs_days_to_full_moon'] = df['days_to_full_moon'].abs()
        return df

    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # --- Keep this method exactly the same as V3 ---
        df = data.copy()
        short = self.periods_short; medium = self.periods_medium; long = self.periods_long
        df['volume_ema_5'] = df['volume'].ewm(span=5, adjust=False).mean()
        df[f'volume_ema_{short}'] = df['volume'].ewm(span=short, adjust=False).mean()
        df[f'volume_ema_{medium}'] = df['volume'].ewm(span=medium, adjust=False).mean()
        df[f'volume_ema_{long}'] = df['volume'].ewm(span=long, adjust=False).mean()
        df['volume_ema_50'] = df['volume'].ewm(span=50, adjust=False).mean()
        df['volume_ema_100'] = df['volume'].ewm(span=100, adjust=False).mean()
        df['volume_rel_ema5'] = df['volume'] / (df['volume_ema_5'] + self.epsilon)
        df[f'volume_rel_ema{medium}'] = df['volume'] / (df[f'volume_ema_{medium}'] + self.epsilon)
        df[f'volume_rel_ema{long}'] = df['volume'] / (df[f'volume_ema_{long}'] + self.epsilon)
        df['vol_ema_ratio_5_20'] = df['volume_ema_5'] / (df[f'volume_ema_{medium}'] + self.epsilon)
        df[f'vol_ema_ratio_{short}_{long}'] = df[f'volume_ema_{short}'] / (df[f'volume_ema_{long}'] + self.epsilon)
        df['vol_ema_ratio_20_100'] = df[f'volume_ema_{medium}'] / (df['volume_ema_100'] + self.epsilon)
        df['volume_change_1d'] = df['volume'].pct_change(1)
        df[f'volume_change_{short}d'] = df['volume'].pct_change(short)
        df[f'volume_change_{medium}d'] = df['volume'].pct_change(medium)
        df['volume_higher'] = (df['volume'] > df['volume'].shift(1)).astype(int)
        df['consecutive_higher_volume'] = self._count_consecutive(df['volume_higher'])
        df['volume_lower'] = (df['volume'] < df['volume'].shift(1)).astype(int)
        df['consecutive_lower_volume'] = self._count_consecutive(df['volume_lower'])
        for p in [short, medium, long]:
             df[f'volume_momentum_{p}'] = self._calculate_momentum(df['volume'], p)
        rolling_vol_std = df['volume'].rolling(window=medium).std()
        df['volume_z_score_ema'] = (df['volume'] - df[f'volume_ema_{long}']) / (rolling_vol_std + self.epsilon)
        df['volume_spike'] = (df['volume_z_score_ema'] > 2).astype(int)
        for p_name, p_val in zip(['5', str(short), str(medium)], [5, short, medium]):
            df[f'volume_ema{p_name}_slope'] = df[f'volume_ema_{p_val}'].diff().fillna(0)
        return df

    def _add_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # --- Keep this method exactly the same as V3 ---
        df = data.copy()
        short = self.periods_short; medium = self.periods_medium; long = self.periods_long
        df['body_size'] = abs(df['close'] - df['open'])
        df['shadow_size'] = df['high'] - df['low']
        df['body_to_shadow_ratio'] = df['body_size'] / (df['shadow_size'] + self.epsilon)
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['shadow_size'] + self.epsilon)
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['shadow_size'] + self.epsilon)
        df['has_long_upper_wick'] = (df['upper_wick'] > 0.6).astype(int)
        df['has_long_lower_wick'] = (df['lower_wick'] > 0.6).astype(int)
        df['return_1d'] = df['close'].pct_change(1)
        df['true_range'] = self._calculate_true_range(df)
        for p in [short, medium, long]:
             df[f'atr_{p}'] = df['true_range'].ewm(span=p, adjust=False).mean()
        for w in self.curve_windows:
            df[f'price_curve_{w}'] = self._calculate_curvature(df['close'], w)
        df['potential_reversal'] = self._detect_potential_reversal(df)
        atr_col = f'atr_{self.periods_medium}'
        df['body_rel_atr'] = df['body_size'] / (df[atr_col] + self.epsilon)
        df['range_rel_atr'] = df['shadow_size'] / (df[atr_col] + self.epsilon)
        return df

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
         # --- Keep this method exactly the same as V3 ---
        df = data.copy()
        short = self.periods_short; medium = self.periods_medium; long = self.periods_long
        df[f'ema_{medium}'] = df['close'].ewm(span=medium, adjust=False).mean()
        df[f'ema_{long}'] = df['close'].ewm(span=long, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df[f'price_vs_ema{medium}'] = df['close'] / (df[f'ema_{medium}'] + self.epsilon)
        df[f'price_vs_ema{long}'] = df['close'] / (df[f'ema_{long}'] + self.epsilon)
        df[f'price_vs_ema200'] = df['close'] / (df['ema_200'] + self.epsilon)
        df[f'ema{medium}_vs_ema{long}'] = df[f'ema_{medium}'] / (df[f'ema_{long}'] + self.epsilon)
        df[f'ema{long}_vs_ema200'] = df[f'ema_{long}'] / (df['ema_200'] + self.epsilon)
        for p in [short, medium, long]:
             df[f'rsi_{p}'] = self._calculate_rsi(df['close'], p)
        if 'true_range' not in df.columns:
            logger.warning("True Range not found, calculating temporarily for ADX.")
            df['true_range'] = self._calculate_true_range(df)
        for p in [short, medium, long]:
             df[f'adx_{p}'] = self._calculate_adx(df, p)
        return df

    # --- Interaction and Explicit Heuristic Features (V4) ---
    def _add_interaction_and_heuristic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """V4: Adds interactions AND explicit heuristic features."""
        df = data.copy()
        medium_atr = f'atr_{self.periods_medium}'
        medium_curve = f'price_curve_{self.periods_medium}' if self.periods_medium in self.curve_windows else f'price_curve_{self.curve_windows[1]}' # Use medium curve

        # --- Standard Interactions (from V3) ---
        req_interactions = ['volume_z_score_ema', 'range_rel_atr', 'body_rel_atr', 'volume_rel_ema5']
        if not all(col in df.columns for col in req_interactions):
            logger.warning(f"Missing required columns for standard interaction features: {req_interactions}. Skipping standard interactions.")
        else:
            df['interaction_vol_range'] = df['volume_z_score_ema'] * (1 / (df['range_rel_atr'] + self.epsilon))
            df['interaction_vol_body'] = df['volume_z_score_ema'] * (1 / (df['body_rel_atr'] + self.epsilon))
            df['interaction_rel_vol_body'] = df['volume_rel_ema5'] * (1 - df['body_rel_atr'].clip(0, 1))

        # --- RSI Divergence (from V3) ---
        req_divergence = [f'rsi_{self.periods_medium}', 'high', 'low']
        if not all(col in df.columns for col in req_divergence):
            logger.warning(f"Missing required columns for RSI divergence: {req_divergence}. Skipping divergence.")
            df['bearish_divergence_rsi'] = 0
            df['bullish_divergence_rsi'] = 0
        else:
            n_divergence = 5
            price_high_n = df['high'].rolling(window=n_divergence).max()
            price_low_n = df['low'].rolling(window=n_divergence).min()
            rsi_high_n = df[f'rsi_{self.periods_medium}'].rolling(window=n_divergence).max()
            rsi_low_n = df[f'rsi_{self.periods_medium}'].rolling(window=n_divergence).min()
            df['bearish_divergence_rsi'] = ((df['high'] >= price_high_n.shift(1)) & (df[f'rsi_{self.periods_medium}'] < rsi_high_n.shift(1))).astype(int)
            df['bullish_divergence_rsi'] = ((df['low'] <= price_low_n.shift(1)) & (df[f'rsi_{self.periods_medium}'] > rsi_low_n.shift(1))).astype(int)

        # --- V4: Short-Term Volume Z-Score ---
        rolling_vol_mean_short = df['volume'].rolling(window=self.short_z_window).mean()
        rolling_vol_std_short = df['volume'].rolling(window=self.short_z_window).std()
        df['volume_z_score_short'] = (df['volume'] - rolling_vol_mean_short) / (rolling_vol_std_short + self.epsilon)
        df['volume_spike_short'] = (df['volume_z_score_short'] > 2).astype(int) # Spike based on short term

        # --- V4: Explicit Heuristic Combination Features ---
        # 1. Volume Spike + Small Body
        req_h1 = ['volume_spike_short', 'body_rel_atr']
        if not all(col in df.columns for col in req_h1):
             logger.warning(f"Missing required columns for Heuristic 1: {req_h1}. Skipping.")
             df['h_vol_spike_small_body'] = 0
        else:
            df['h_vol_spike_small_body'] = ((df['volume_spike_short'] == 1) & (df['body_rel_atr'] < self.small_body_threshold)).astype(int)

        # 2. Volume Exhaustion + Price Curve
        req_h2 = ['consecutive_lower_volume', medium_curve]
        if not all(col in df.columns for col in req_h2):
            logger.warning(f"Missing required columns for Heuristic 2: {req_h2}. Skipping.")
            df['h_vol_exhaust_bull_curve'] = 0
            df['h_vol_exhaust_bear_curve'] = 0
        else:
            df['h_vol_exhaust_bull_curve'] = ((df['consecutive_lower_volume'] >= self.vol_exhaust_days) & (df[medium_curve] > self.price_curve_threshold)).astype(int)
            df['h_vol_exhaust_bear_curve'] = ((df['consecutive_lower_volume'] >= self.vol_exhaust_days) & (df[medium_curve] < -self.price_curve_threshold)).astype(int)

        # 3. Reversal Pattern Near Full Moon
        req_h3 = ['potential_reversal', 'full_moon_window_3d']
        if not all(col in df.columns for col in req_h3):
            logger.warning(f"Missing required columns for Heuristic 3: {req_h3}. Skipping.")
            df['h_reversal_near_full_moon'] = 0
        else:
            # Combine: 1 if bullish reversal near moon, -1 if bearish reversal near moon, 0 otherwise
            df['h_reversal_near_full_moon'] = df['potential_reversal'] * df['full_moon_window_3d']


        # Fill NaNs created by new rolling calculations (e.g., short Z-score)
        # Do this *before* returning, but after all calculations dependent on rolling windows
        cols_to_check_nan = ['volume_z_score_short','h_vol_spike_small_body','h_vol_exhaust_bull_curve', 'h_vol_exhaust_bear_curve', 'h_reversal_near_full_moon']
        for col in cols_to_check_nan:
             if col in df.columns:
                  df[col] = df[col].fillna(0)

        return df

    # --- Helper Methods (Keep V3 versions - already parameterized) ---
    def _add_target_variable(self, data: pd.DataFrame, turning_points: List[Dict]) -> pd.DataFrame:
        # --- Keep this method exactly the same as V3 ---
        df = data.copy(); df['is_turning_point'] = 0.0; df['is_top'] = 0.0; df['is_bottom'] = 0.0
        target_tz = df.index.tz; known_tp_norm_dates = {}; parse_errors = 0; invalid_types = 0
        for tp in turning_points:
            try:
                date_input = tp.get('date')
                if date_input is None: parse_errors += 1; continue
                date_raw = pd.Timestamp(date_input); tp_type = tp.get('type', 'unknown').lower()
                if tp_type not in ['top', 'high', 'bottom', 'low']: invalid_types += 1; continue
                if target_tz: date_final = date_raw.tz_localize(target_tz) if date_raw.tzinfo is None else date_raw.tz_convert(target_tz)
                else: date_final = date_raw.tz_localize(None) if date_raw.tzinfo is not None else date_raw
                date_normalized = date_final.normalize()
                known_tp_norm_dates[date_normalized] = 'top' if tp_type in ['top', 'high'] else 'bottom'
            except Exception as e: parse_errors += 1; logger.warning(f"TP parse error: {e}")
        if parse_errors > 0: logger.warning(f"TP date parse errors: {parse_errors}")
        if invalid_types > 0: logger.warning(f"Skipped invalid TP types: {invalid_types}")
        matched_dates = df.index.intersection(known_tp_norm_dates.keys())
        if not matched_dates.empty:
            logger.info(f"Assigning labels for {len(matched_dates)} matched TPs.")
            df.loc[matched_dates, 'is_turning_point'] = 1.0
            for date in matched_dates:
                tp_type = known_tp_norm_dates[date]
                if tp_type == 'top': df.loc[date, 'is_top'] = 1.0
                elif tp_type == 'bottom': df.loc[date, 'is_bottom'] = 1.0
        else: logger.warning("No known TPs found in market data index.")
        window_size = 1
        if window_size > 0 and not matched_dates.empty:
            df['is_orig_top'] = df['is_top']; df['is_orig_bottom'] = df['is_bottom']
            for date in matched_dates:
                 is_orig_top = df.loc[date, 'is_orig_top'] == 1.0; is_orig_bottom = df.loc[date, 'is_orig_bottom'] == 1.0
                 for offset in range(-window_size, window_size + 1):
                     if offset == 0: continue
                     nearby_date = date + pd.Timedelta(days=offset)
                     if nearby_date in df.index and df.loc[nearby_date, 'is_turning_point'] != 1.0:
                         df.loc[nearby_date, 'is_turning_point'] = max(df.loc[nearby_date, 'is_turning_point'], 0.5)
                         if is_orig_top and df.loc[nearby_date, 'is_orig_top'] != 1.0: df.loc[nearby_date, 'is_top'] = max(df.loc[nearby_date, 'is_top'], 0.5)
                         if is_orig_bottom and df.loc[nearby_date, 'is_orig_bottom'] != 1.0: df.loc[nearby_date, 'is_bottom'] = max(df.loc[nearby_date, 'is_bottom'], 0.5)
            df = df.drop(columns=['is_orig_top', 'is_orig_bottom'])
        return df

    def _count_consecutive(self, series: pd.Series) -> pd.Series: return series * (series.groupby((series != series.shift()).cumsum()).cumcount() + 1)
    def _calculate_momentum(self, series: pd.Series, period: int) -> pd.Series: shifted_series = series.shift(period); momentum = series / (shifted_series + self.epsilon) - 1; return momentum.fillna(0)
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series: high_low = data['high'] - data['low']; high_close_prev = abs(data['high'] - data['close'].shift(1)); low_close_prev = abs(data['low'] - data['close'].shift(1)); ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1); true_range = ranges.max(axis=1, skipna=False); return true_range
    def _calculate_curvature(self, series: pd.Series, window: int) -> pd.Series: dx1 = series.diff(1); dx2 = dx1.diff(1); min_p = max(1, window // 2 + 1); smoothed_dx2 = dx2.rolling(window=window, min_periods=min_p).mean(); avg_price = series.rolling(window=window, min_periods=min_p).mean(); curvature = smoothed_dx2 / (avg_price + self.epsilon); return curvature.fillna(0)
    def _detect_potential_reversal(self, data: pd.DataFrame) -> pd.Series:
        if 'return_1d' not in data.columns: df = data.copy(); df['return_1d'] = df['close'].pct_change(1).fillna(0)
        else: df = data
        if 'has_long_lower_wick' not in df.columns or 'has_long_upper_wick' not in df.columns: return pd.Series(0, index=df.index)
        result = pd.Series(0, index=df.index); prev_return = df['return_1d'].shift(1).fillna(0); bullish_reversal = ((prev_return < -0.005) & (df['close'] > df['open']) & (df['has_long_lower_wick'] == 1)); bearish_reversal = ((prev_return > 0.005) & (df['close'] < df['open']) & (df['has_long_upper_wick'] == 1)); result[bullish_reversal] = 1; result[bearish_reversal] = -1; return result
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series: delta = prices.diff(); gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0); avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean(); rs = avg_gain / (avg_loss + self.epsilon); rsi = 100.0 - (100.0 / (1.0 + rs)); rsi = rsi.fillna(50); return rsi
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        if 'true_range' not in data.columns: raise ValueError("ADX calculation requires 'true_range' column.")
        atr = data['true_range'].ewm(span=period, adjust=False, min_periods=period).mean(); high_diff = data['high'].diff(); low_diff = data['low'].diff(); plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0); minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0); plus_dm_ema = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean(); minus_dm_ema = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean(); plus_di = 100 * (plus_dm_ema / (atr + self.epsilon)); minus_di = 100 * (minus_dm_ema / (atr + self.epsilon)); dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + self.epsilon); adx = dx.ewm(span=period, adjust=False, min_periods=period).mean(); adx = adx.fillna(20); return adx

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    logger.info("Running FeatureEngineer example (V4 - Explicit Heuristics)...")
    try:
        import yfinance as yf; symbol = "SPY"; data = yf.download(symbol, start="2019-01-01", end="2023-01-01")
        data.columns = [c.lower() for c in data.columns]; engineer = FeatureEngineer()
        features = engineer.create_features(data);
        print(f"\nGenerated {len(features.columns)} features for {len(features)} days (after NaN drop)")
        print("\nSample features (V4):"); print(features.tail())
        print("\nColumns:", features.columns.tolist()); print(f"\nNumber of features: {len(features.columns)}")
        print("\nNew Heuristic Features Sample:")
        print(features[['volume_z_score_short', 'h_vol_spike_small_body', 'h_vol_exhaust_bull_curve', 'h_vol_exhaust_bear_curve', 'h_reversal_near_full_moon']].tail(20))
    except ImportError: logger.error("Please install yfinance (`pip install yfinance`) to run the example.")
    except Exception as e: logger.error(f"An error occurred during the example: {e}", exc_info=True)

# --- END OF feature_engineering.py (V4 - Explicit Heuristics) ---