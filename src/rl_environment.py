# --- START OF src/rl_environment.py (v2 - Absolute Import) ---

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Import base classes
import joblib

# --- Use Absolute Import --- MODIFIED
# Assumes 'src' directory is in sys.path (handled by train_rl.py)
from feature_engineering import FeatureEngineer
# --- End Absolute Import ---

logger = logging.getLogger(__name__)

class MarketEnvironment:

    def __init__(self,
                 market_data: pd.DataFrame,
                 turning_points: List[Dict],
                 feature_engineer: FeatureEngineer,
                 sequence_length: int,
                 episode_max_steps: int = 500,
                 reward_lookahead_k: int = 2,
                 reward_config: Dict = None,
                 initial_offset_range: Tuple[int, int] = (0, 252),
                 debug: bool = False,
                 scaler: Optional[Any] = None):

        if not isinstance(market_data.index, pd.DatetimeIndex):
            raise ValueError("market_data must have a DatetimeIndex.")

        self.feature_engineer = feature_engineer
        self.sequence_length = sequence_length
        self.k = reward_lookahead_k
        self.episode_max_steps = episode_max_steps
        self.debug = debug
        self.initial_offset_min, self.initial_offset_max = initial_offset_range
        self.scaler = scaler

        # --- Reward Configuration ---
        self.reward_config = {
            'hit_top': 10.0, 'hit_bottom': 10.0, 'close_hit_decay': 0.6,
            'wrong_direction': -7.0, 'false_positive': -5.0, 'missed_tp_penalty': -4.0,
            'missed_close_tp_decay': 0.7, 'correct_inaction': 0.1
        }
        if reward_config: self.reward_config.update(reward_config)

        # --- Preprocess Data ---
        logger.info("Preprocessing data and features for RL environment...")
        self._market_data_full = market_data.copy()
        # Generate features once (using the provided FeatureEngineer instance)
        features_df_raw = self.feature_engineer.create_features(
            self._market_data_full, include_target=False # Target not needed for state features
        )

        # Select only numeric features for state representation
        features_df_numeric = features_df_raw.select_dtypes(include=np.number)
        self._feature_names = features_df_numeric.columns.tolist()
        logger.info(f"Selected {len(self._feature_names)} numeric features for state.")

        # Convert numeric data to numpy array and handle NaNs
        self._feature_values = features_df_numeric.values.astype(np.float32)
        if np.isnan(self._feature_values).any():
             logger.warning("NaNs detected in numeric features BEFORE scaling. Filling with 0.")
             self._feature_values = np.nan_to_num(self._feature_values) # Fill NaNs with 0

        self._dates = features_df_numeric.index # Keep track of dates corresponding to features
        self._num_features = self._feature_values.shape[1]
        self._total_steps = len(self._feature_values) # Total number of time steps available

        # --- Log Scaler Info ---
        if self.scaler:
            logger.info(f"Scaler provided: {type(self.scaler)}")
            # Verify scaler compatibility
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self._num_features:
                 logger.error(f"Scaler feature mismatch! Scaler expects {self.scaler.n_features_in_}, data has {self._num_features}.")
                 raise ValueError("Scaler feature count does not match data feature count.")
            # Apply scaling once to the entire feature set if scaler exists
            logger.info("Applying scaling to all feature values...")
            try:
                self._feature_values = self.scaler.transform(self._feature_values)
                logger.info("Feature values scaled successfully.")
                if np.isnan(self._feature_values).any():
                     logger.warning("NaNs detected in features AFTER scaling. Check scaler or input data.")
                     # Optional: Fill NaNs again if scaling introduces them somehow
                     # self._feature_values = np.nan_to_num(self._feature_values)
            except Exception as e:
                 logger.error(f"Error applying scaler transform to all features: {e}", exc_info=True)
                 raise e # Fail fast if scaling fails
        else:
             logger.warning("No scaler provided to environment. State features will NOT be scaled.")


        # --- Process True Turning Points ---
        self._true_tp = {} # Dictionary mapping date -> 'top' or 'bottom'
        for tp in turning_points:
             try:
                date_input = tp.get('date'); tp_type = tp.get('type', 'unknown').lower()
                if date_input is None or tp_type not in ['top', 'high', 'bottom', 'low']: continue
                # Normalize date and check if it exists in our feature data's index
                date = pd.Timestamp(date_input).normalize()
                if date in self._dates:
                    self._true_tp[date] = 'top' if tp_type in ['top', 'high'] else 'bottom'
             except Exception as e: logger.warning(f"Could not process turning point {tp}: {e}")
        logger.info(f"Processed {len(self._true_tp)} true turning points mapped to available dates.")

        # --- Action/Observation Space ---
        self.action_space_n = 3 # 0: Hold, 1: Predict Top, 2: Predict Bottom
        self.observation_shape = (self.sequence_length, self._num_features)

        # --- Episode State ---
        self._current_step = 0 # Index of the *last* day in the current observation sequence
        self._episode_start_step = 0 # Index of the first possible step for this episode
        self._episode_steps_taken = 0 # Steps taken within the current episode

        logger.info("RL Environment Initialized.")


    def reset(self) -> np.ndarray:
        """Resets the environment to a new starting point for an episode."""
        # Determine valid range for the start of an episode sequence
        # Need space for sequence_length, max_episode_steps, and lookahead_k
        min_required_steps = self.sequence_length + self.episode_max_steps + self.k + 1
        max_start_index = self._total_steps - min_required_steps

        # Ensure initial offset range is within valid bounds
        safe_min_offset = max(0, min(self.initial_offset_min, max_start_index))
        safe_max_offset = max(safe_min_offset, min(self.initial_offset_max, max_start_index))

        if safe_max_offset <= safe_min_offset:
            self._episode_start_step = safe_min_offset
        else:
            self._episode_start_step = np.random.randint(safe_min_offset, safe_max_offset + 1) # +1 because randint is exclusive upper

        # Set the current step to be the end of the first sequence
        self._current_step = self._episode_start_step + self.sequence_length - 1
        self._episode_steps_taken = 0

        if self.debug:
            start_date = self._dates[self._episode_start_step]
            current_date = self._dates[self._current_step]
            logger.debug(f"Resetting episode. Start Step Index: {self._episode_start_step} ({start_date.date()}), Current Step Index: {self._current_step} ({current_date.date()})")

        return self._get_state() # Return the initial state observation


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advances the environment by one time step based on the action."""
        if not (0 <= action < self.action_space_n): raise ValueError(f"Invalid action: {action}.")

        current_date = self._dates[self._current_step] # Date corresponding to the *end* of the sequence just observed

        # Calculate reward based on the action taken *at* current_step_idx
        reward = self._calculate_reward(action, self._current_step)

        # Advance time by one step
        self._current_step += 1
        self._episode_steps_taken += 1

        # Check for termination conditions
        done = False
        if self._episode_steps_taken >= self.episode_max_steps:
            done = True
            if self.debug: logger.debug(f"Episode ended: Max steps ({self.episode_max_steps}) reached.")
        # Ensure we have enough future data for reward calculation lookahead
        if self._current_step >= self._total_steps - self.k -1:
            done = True
            if self.debug: logger.debug(f"Episode ended: Not enough future data (step {self._current_step}, total {self._total_steps}, k {self.k}).")

        # Get the state observation for the *next* step
        next_state = self._get_state()

        # Info dictionary (can add more diagnostics if needed)
        info = {'date': self._dates[self._current_step] if not done else current_date}

        if self.debug and self._episode_steps_taken % 50 == 0:
            logger.debug(f"Step {self._episode_steps_taken}: Date {current_date.date()}, Action {action}, Reward {reward:.2f}, Done {done}")

        return next_state, reward, done, info


    def _get_state(self) -> np.ndarray:
        """Extracts the feature sequence ending at the current step."""
        # Calculate start and end indices for the sequence
        # End index is inclusive for slicing numpy arrays
        end_idx = self._current_step + 1
        start_idx = end_idx - self.sequence_length

        # Handle boundary condition at the beginning of data
        if start_idx < 0:
            # This shouldn't happen if reset() logic is correct, but handle defensively
            logger.warning(f"Attempting to get state with start_idx < 0 ({start_idx}). Reset logic might be flawed.")
            padding_needed = abs(start_idx)
            valid_data = self._feature_values[0:end_idx]
            # Pad with zeros at the beginning
            padding = np.zeros((padding_needed, self._num_features), dtype=np.float32)
            state_data = np.vstack((padding, valid_data))
        else:
            # Extract the sequence
             # Note: We are using the already-scaled self._feature_values if scaler was provided
            state_data = self._feature_values[start_idx:end_idx]

        # Verify shape
        if state_data.shape != self.observation_shape:
            # This might happen if slicing near boundaries goes wrong or data length is too short
            logger.error(f"State shape mismatch! Expected {self.observation_shape}, got {state_data.shape}. Current step: {self._current_step}, Start: {start_idx}, End: {end_idx}")
            # Attempt to pad or truncate, but this indicates a deeper issue
            # For now, raise error might be better
            raise ValueError(f"State shape error. Expected {self.observation_shape}, got {state_data.shape}")

        # --- Scaling is already applied to self._feature_values during __init__ ---
        # No scaling needed here if applied upfront.

        return state_data


    def _get_true_tp_type(self, step_index: int) -> Optional[str]:
        """Checks if a true turning point exists at the given step index."""
        if 0 <= step_index < len(self._dates):
            date = self._dates[step_index]
            return self._true_tp.get(date, None) # Return 'top', 'bottom', or None
        return None

    def _calculate_reward(self, action: int, current_step_idx: int) -> float:
        """Calculates the reward for taking 'action' at 'current_step_idx'."""
        # Determine the prediction type based on action
        predicted_type = None
        if action == 1: predicted_type = 'top'    # Agent predicts a Top
        elif action == 2: predicted_type = 'bottom' # Agent predicts a Bottom
        # action == 0 means Hold (no prediction)

        reward = 0.0
        hit_found_in_window = False

        # Look ahead k steps (including the current step, so k+1 checks)
        for j in range(self.k + 1):
            step_to_check = current_step_idx + j
            true_type = self._get_true_tp_type(step_to_check)

            # If agent predicted a Top (action 1)
            if predicted_type == 'top':
                if true_type == 'top': # Correct prediction!
                    decay_factor = self.reward_config.get('close_hit_decay', 1.0) ** j
                    reward += self.reward_config['hit_top'] * decay_factor
                    hit_found_in_window = True
                    break # Stop checking window on first correct hit
                elif true_type == 'bottom': # Wrong direction prediction
                    reward += self.reward_config['wrong_direction']
                    hit_found_in_window = True # Penalized, but counts as TP found
                    break # Stop checking window

            # If agent predicted a Bottom (action 2)
            elif predicted_type == 'bottom':
                if true_type == 'bottom': # Correct prediction!
                    decay_factor = self.reward_config.get('close_hit_decay', 1.0) ** j
                    reward += self.reward_config['hit_bottom'] * decay_factor
                    hit_found_in_window = True
                    break # Stop checking window
                elif true_type == 'top': # Wrong direction prediction
                    reward += self.reward_config['wrong_direction']
                    hit_found_in_window = True # Penalized, but counts as TP found
                    break # Stop checking window

            # If agent predicted Hold (action 0)
            elif predicted_type is None:
                if true_type is not None: # Agent held but missed a TP
                    decay_factor = self.reward_config.get('missed_close_tp_decay', 1.0) ** j
                    reward += self.reward_config['missed_tp_penalty'] * decay_factor
                    hit_found_in_window = True # Count as TP found (missed)
                    # Decide if we should break here or penalize for all TPs in window?
                    # Current: penalize for the first missed TP found in window.

        # --- Penalties/Rewards outside the lookahead loop ---
        # Penalty for predicting a TP when none occurred in the window
        if predicted_type is not None and not hit_found_in_window:
            reward += self.reward_config['false_positive']

        # Reward for correctly holding when no TP occurred in the window
        elif predicted_type is None and not hit_found_in_window:
            reward += self.reward_config['correct_inaction']

        return reward

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names used in the state."""
        return self._feature_names


# --- Example Usage ---
if __name__ == '__main__':
     logging.basicConfig(level=logging.DEBUG)
     logger.info("Running RL Environment example...")

     # Create dummy data
     dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
     data = pd.DataFrame({
         'open': np.random.rand(200) * 10 + 100,
         'high': np.random.rand(200) * 5 + 105,
         'low': np.random.rand(200) * 5 + 95,
         'close': np.random.rand(200) * 10 + 100,
         'volume': np.random.randint(10000, 50000, size=200)
     }, index=dates)
     data['high'] = data[['open', 'close']].max(axis=1) + np.random.rand(200) * 3
     data['low'] = data[['open', 'close']].min(axis=1) - np.random.rand(200) * 3

     # Create dummy turning points
     tps = [
         {'date': dates[50], 'type': 'top'},
         {'date': dates[100], 'type': 'bottom'},
         {'date': dates[150], 'type': 'top'}
     ]

     # Create dummy feature engineer and lunar calculator
     class DummyLunar:
         def calculate_lunar_phase(self, date): return {'phase_percentage': 50.0, 'phase_name': 'Full', 'days_to_full_moon': 0, 'is_near_full_moon': True}
     lunar_calc = DummyLunar()
     # Use the actual FeatureEngineer
     fe = FeatureEngineer(lunar_calculator=lunar_calc)

     # Create dummy scaler
     dummy_features = fe.create_features(data, include_target=False).select_dtypes(include=np.number)
     dummy_scaler = MinMaxScaler().fit(np.nan_to_num(dummy_features.values))
     logger.info(f"Dummy scaler fitted on {dummy_scaler.n_features_in_} features.")


     # Initialize environment
     seq_len = 20
     env = MarketEnvironment(data, tps, fe, seq_len, debug=True, scaler=dummy_scaler, episode_max_steps=50, initial_offset_range=(0, 50))

     # Run a sample episode
     logger.info("\n--- Running Sample Episode ---")
     state = env.reset()
     done = False
     total_reward = 0
     step_count = 0
     while not done:
         # Choose a random action (0, 1, or 2)
         action = np.random.randint(0, env.action_space_n)
         next_state, reward, done, info = env.step(action)
         logger.info(f"Step {step_count+1}: Action={action}, Reward={reward:.3f}, Date={info['date'].date()}, Done={done}")
         # Check state shape and range (should be 0-1 if scaled)
         if state.shape != env.observation_shape: logger.error(f"State shape error: {state.shape}")
         if env.scaler and (np.min(state) < -0.01 or np.max(state) > 1.01): logger.warning(f"State out of range [{np.min(state):.2f}, {np.max(state):.2f}]")

         state = next_state
         total_reward += reward
         step_count += 1

     logger.info(f"Episode finished after {step_count} steps. Total reward: {total_reward:.3f}")
     logger.info("--- End Sample Episode ---")

# --- END OF src/rl_environment.py (v2 - Absolute Import) ---