# --- START OF train_rl.py (v29 - Disabled CheckNumerics) ---

import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
import tensorflow as tf
# --- Scaling Imports ---
from sklearn.preprocessing import MinMaxScaler
import joblib
# --- End Scaling Imports ---

# --- Setup Logging and Paths EARLY ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TensorFlow Debugging (Optional) ---
# tf.debugging.enable_check_numerics() # <<< DISABLED - Caused XLA incompatibility error
# logger.warning("TensorFlow CheckNumerics ENABLED.") # <<< DISABLED
# --- End Debugging Setup ---

# --- Memory Monitoring (Optional but Recommended) ---
try:
    import psutil # For system RAM monitoring
    import nvidia_smi # For GPU VRAM monitoring (install: pip install nvidia-ml-py3)
    nvidia_smi.nvmlInit()
    gpu_device_index = 0 # Assuming you are using the first GPU (GPU:0)
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_device_index)
    logger.info("psutil and nvidia-ml-py3 found. Will log memory usage.")
    MEMORY_LOGGING_ENABLED = True
except (ImportError, Exception) as e:
    logger.warning(f"Could not import psutil or nvidia-ml-py3/initialize NVML: {e}. Memory usage won't be logged.")
    MEMORY_LOGGING_ENABLED = False
# --- End Memory Monitoring Setup ---


import sys
script_path = Path(__file__).resolve()
project_root = script_path.parent
src_dir = project_root / 'src'
if not src_dir.is_dir():
    project_root = script_path.parent.parent; src_dir = project_root / 'src'
    if not src_dir.is_dir():
        project_root = Path.cwd(); src_dir = project_root / 'src'
        if not src_dir.is_dir(): logger.error(f"CRITICAL: Cannot find 'src' directory."); exit(1)
logger.info(f"Project Root: {project_root}"); logger.info(f"Source Directory: {src_dir}")
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path: sys.path.insert(0, str(src_dir))
logger.debug(f"sys.path after adjustment: {sys.path}")

# --- Now perform imports ---
try:
    from src.feature_engineering import FeatureEngineer
    from src.lunar_phase_calculator import LunarPhaseCalculator
    from src.rl_environment import MarketEnvironment
    # Use D3QN Agent (expecting v6+)
    from src.d3qn_agent import D3QNAgent
    # Import custom layers if they exist and are needed by the agent build
    try:
        from train_model import PositionalEmbedding, TransformerBlock
    except ImportError:
        logger.warning("train_model.py or its custom layers not found. Agent might use dummies.")

except (ModuleNotFoundError, ImportError) as e:
    logger.error(f"Error importing modules: {e}", exc_info=True); exit(1)
# --- END IMPORTS ---


def log_memory_usage():
    """Logs current system RAM and GPU VRAM usage."""
    if not MEMORY_LOGGING_ENABLED: return
    try:
        # System RAM
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        # GPU VRAM
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        vram_used_gb = mem_info.used / (1024**3)
        vram_total_gb = mem_info.total / (1024**3)
        logger.debug(f"Memory - RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not log memory usage: {e}")


def plot_learning_curve(scores, filename, window=100):
    """Plots the moving average of scores."""
    if not scores or len(scores) < window: logger.warning(f"Not enough scores ({len(scores)}) for plot."); return
    try:
        scores_numeric = [s if isinstance(s, (int, float)) and not np.isnan(s) else 0 for s in scores]
        running_avg = np.convolve(scores_numeric, np.ones(window)/window, mode='valid')
        if len(running_avg) == 0: logger.warning("No running average data. Skipping plot."); return
        plt.figure(figsize=(10, 6)); plt.plot(np.arange(window - 1, len(scores_numeric)), running_avg)
        plt.title(f'Reward Moving Average ({window} episodes)'); plt.xlabel('Episode'); plt.ylabel('Average Reward')
        plt.grid(True); plt.savefig(filename); plt.close(); logger.info(f"Learning curve saved: {filename}")
    except Exception as e: logger.error(f"Plot generation failed: {e}", exc_info=True)


def main():
    parser = ArgumentParser(description="Train D3QN RL Agent")
    # Arguments (keep as before, remove --run-eagerly)
    parser.add_argument("--config-file", type=str, default="config.json", help="Config JSON path")
    parser.add_argument("--data-file", type=str, required=True, help="Market data CSV path")
    parser.add_argument("--turning-points-file", type=str, required=True, help="Turning points CSV path")
    parser.add_argument("--output-dir", type=str, default="rl_output_d3qn", help="Output directory path")
    parser.add_argument("--load-model-dir", type=str, default=None, help="Load pre-trained agent dir")
    parser.add_argument("--load-model-prefix", type=str, default=None, help="Load model filename prefix")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--debug", action="store_true", help="Enable env debug logging")
    parser.add_argument("--scaler-fit-ratio", type=float, default=0.8, help="Scaler fit data ratio")
    parser.add_argument("--force-refit-scaler", action="store_true", help="Force scaler refit")
    args = parser.parse_args()

    # --- Load Config ---
    config_path = Path(args.config_file); logger.info(f"Loading config: {config_path}")
    if not config_path.is_file(): logger.error(f"Config not found: {config_path}"); exit(1)
    try: config = json.load(open(config_path, 'r'))
    except Exception as e: logger.error(f"Error loading config: {e}"); exit(1)
    model_train_config = config.get('model_training', {}); d3qn_config = config.get('d3qn_training', {}); rl_env_config = config.get('rl_training', {})
    if not all([model_train_config, d3qn_config, rl_env_config]): logger.error("Config needs model_training, d3qn_training, rl_training."); exit(1)

    # --- Setup Output Directory ---
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S'); run_name = f"d3qn_agent_{timestamp}"
    model_save_prefix_template = f"{run_name}_episode_{{}}"; final_model_save_prefix = f"{run_name}_final"
    results_save_path = output_dir / f"training_log_{run_name}.csv"; plot_save_path = output_dir / f"learning_curve_{run_name}.png"
    scaler_save_path = output_dir / f"state_scaler_{timestamp}.joblib"

    # --- Load Data & TPs (Keep as before) ---
    data_path = Path(args.data_file); logger.info(f"Loading data: {data_path}")
    # ... (keep data loading logic) ...
    try: full_market_data = pd.read_csv(data_path, index_col=0, parse_dates=True); full_market_data.columns = map(str.lower, full_market_data.columns); logger.info(f"Loaded market data: {len(full_market_data)} rows")
    except Exception as e: logger.error(f"Error loading data: {e}", exc_info=True); exit(1)
    tp_path = Path(args.turning_points_file); logger.info(f"Loading TPs: {tp_path}")
    # ... (keep TP loading logic) ...
    try: tp_df = pd.read_csv(tp_path); tp_df.columns = map(str.lower, tp_df.columns); date_col = next((c for c in ['date','time','datetime'] if c in tp_df.columns), None); type_col = next((c for c in ['type','direction'] if c in tp_df.columns), None); assert date_col and type_col; tp_df[date_col] = pd.to_datetime(tp_df[date_col]); turning_points = tp_df[[date_col, type_col]].rename(columns={date_col:'date', type_col:'type'}).to_dict('records'); logger.info(f"Loaded {len(turning_points)} TPs.")
    except Exception as e: logger.error(f"Error loading TPs: {e}", exc_info=True); exit(1)


    # --- Initialize Feature Engineer ---
    lunar_calc = LunarPhaseCalculator(); feature_engineer = FeatureEngineer(lunar_calculator=lunar_calc)

    # --- Fit/Load Scaler (Keep as before) ---
    scaler = None
    scaler_file_path = output_dir / "state_scaler.joblib" # Use a fixed name for reuse
    if scaler_file_path.exists() and not args.force_refit_scaler:
        try:
            logger.info(f"Loading existing scaler: {scaler_file_path}")
            scaler = joblib.load(scaler_file_path)
        except Exception as e:
            logger.error(f"Error loading scaler: {e}. Refitting.", exc_info=True)
            scaler = None # Ensure scaler is None if loading failed
    
    if scaler is None:
        logger.info(f"Fitting scaler (Ratio: {args.scaler_fit_ratio*100:.0f}%)...")
        try:
            fit_split = int(len(full_market_data) * args.scaler_fit_ratio)
            seq_len_temp = model_train_config.get('model_params', {}).get('sequence_length', 60)
            assert fit_split >= seq_len_temp, f"Fit rows ({fit_split}) < seq len ({seq_len_temp})."
            data_fit = full_market_data.iloc[:fit_split]
            features_fit_raw = feature_engineer.create_features(data_fit, include_target=False)
            features_fit_num = features_fit_raw.select_dtypes(include=np.number)
            assert not features_fit_num.empty, "No numeric features found for scaler fitting."
            num_vals = features_fit_num.values.astype(np.float32)
            num_vals = np.nan_to_num(num_vals) # Handle NaNs before fitting
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(num_vals)
            logger.info(f"Scaler fitted on {num_vals.shape[1]} features.")
            joblib.dump(scaler, scaler_file_path) # Save with fixed name
            logger.info(f"Scaler saved: {scaler_file_path}")
            del data_fit, features_fit_raw, features_fit_num, num_vals # Clean up memory
        except Exception as e:
            logger.error(f"Scaler fitting error: {e}", exc_info=True)
            exit(1)


    # --- Initialize Components ---
    logger.info("Initializing RL components...")
    try:
        seq_len = model_train_config.get('model_params', {}).get('sequence_length', 60)
        env = MarketEnvironment(market_data=full_market_data, turning_points=turning_points,
            feature_engineer=feature_engineer, sequence_length=seq_len,
            episode_max_steps=rl_env_config.get('max_episode_steps', 1000), # Increased from 500 in config
            reward_lookahead_k=rl_env_config.get('reward_lookahead_k', 2),
            reward_config=rl_env_config.get('reward_config', None),
            initial_offset_range=tuple(rl_env_config.get('initial_offset_range', [0, 500])), # Increased from [0, 252] in config
            debug=args.debug, scaler=scaler) # Pass the scaler here

        # --- Instantiate Optimizer ---
        lr=d3qn_config.get('learning_rate', 1e-4); clip=d3qn_config.get('gradient_clipnorm', 1.0); opt_type=d3qn_config.get('optimizer', 'AdamW').lower(); wd=d3qn_config.get('weight_decay', 1e-4)
        if opt_type == 'adamw': optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=clip, name='DQN_AdamW'); logger.info(f"Using AdamW. LR={lr}, WD={wd}, Clip={clip}")
        else: optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clip, name='DQN_Adam'); logger.info(f"Using Adam. LR={lr}, Clip={clip}")

        # --- Instantiate Agent ---
        agent = D3QNAgent(state_shape=env.observation_shape, action_space_n=env.action_space_n,
            optimizer=optimizer, model_params_config=model_train_config, rl_params_config=d3qn_config)

    except Exception as e: logger.error(f"Component init error: {e}", exc_info=True); exit(1)

    # --- Load Model Weights ---
    if args.load_model_dir and args.load_model_prefix:
        load_dir=Path(args.load_model_dir); load_prefix=args.load_model_prefix; logger.info(f"Attempting load: {load_dir} / {load_prefix}")
        # Check if agent loaded successfully before assuming it's ready
        if not agent.load_model(load_dir, load_prefix):
             logger.warning(f"Failed load. Starting fresh.")
             # Explicitly reset epsilon if load fails and you want to start fresh
             agent.epsilon = agent.rl_params.get('epsilon_start', 1.0)
             agent.train_step_counter = 0 # Reset counter if starting fresh
             logger.info("Resetting epsilon and step counter for fresh start.")
    else: logger.info("No model loading specified. Starting fresh.")


    # --- Training Loop ---
    logger.info(f"Starting D3QN training: {args.num_episodes} episodes...")
    scores_history, episode_lengths, training_log = [], [], []
    log_interval=d3qn_config.get("log_interval", 50); save_interval=d3qn_config.get("save_interval", 200)
    memory_log_interval = 5000 # <<< How often to log memory usage (in steps)
    total_steps_collected, start_time = 0, time.time()
    last_memory_log_step = 0

    # --- Resume Counters if Model Loaded ---
    # Note: Need to store/load these counters for proper resuming
    # Placeholder: Assume starting from 0 if fresh, or potentially load from a state file if implemented
    if agent.train_step_counter > 0:
         logger.info(f"Resuming training from step {agent.train_step_counter}")
         total_steps_collected = agent.train_step_counter # Initialize total steps based on loaded model state if possible
         # Recalculate epsilon based on resumed step counter
         if agent.epsilon_decay_steps > 0:
              steps_decayed = min(agent.train_step_counter, agent.epsilon_decay_steps)
              agent.epsilon = agent.rl_params.get('epsilon_start', 1.0) - steps_decayed * agent.epsilon_delta
              agent.epsilon = max(agent.epsilon_min, agent.epsilon)
         logger.info(f"Adjusted epsilon based on loaded steps: {agent.epsilon:.4f}")

    try:
        for episode in range(args.num_episodes):
            state = env.reset(); log_memory_usage() # Log memory at start of episode
            done, score, steps_in_episode = False, 0, 0
            episode_loss_sum, episode_q_sum, learn_steps = 0.0, 0.0, 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state; total_steps_collected += 1; steps_in_episode += 1; score += reward

                # --- Learning Step ---
                if total_steps_collected >= agent.learning_starts:
                    # logger.debug(f"Step {total_steps_collected}: Calling agent.learn()") # DEBUG
                    loss, avg_q = agent.learn()
                    # logger.debug(f"Step {total_steps_collected}: learn() returned loss={loss}, avg_q={avg_q}") # DEBUG
                    if not np.isnan(loss):
                        episode_loss_sum += loss; learn_steps += 1
                    if not np.isnan(avg_q):
                         episode_q_sum += avg_q
                    # Check immediately if learn returned NaN - indicates gradient problem
                    if np.isnan(loss):
                         logger.error(f"!!! NaN Loss detected at step {total_steps_collected} in episode {episode+1} !!!")
                         # Optional: Add more debugging here, like printing state/next_state summaries
                         # state_summary = np.mean(state, axis=0) # Example: Mean across seq len
                         # next_state_summary = np.mean(next_state, axis=0)
                         # logger.error(f"State mean summary:\n{state_summary}")
                         # logger.error(f"Next state mean summary:\n{next_state_summary}")
                         # Optional: Could raise an exception to stop training
                         # raise RuntimeError("NaN Loss encountered during training.")
                    # Check if process hangs after learn() (less likely now)
                    # logger.debug(f"Step {total_steps_collected}: learn() finished.") # DEBUG

                # Log memory periodically based on total steps
                if MEMORY_LOGGING_ENABLED and (total_steps_collected - last_memory_log_step) >= memory_log_interval:
                    log_memory_usage()
                    last_memory_log_step = total_steps_collected

            # End of episode
            scores_history.append(score); episode_lengths.append(steps_in_episode)
            avg_score = np.nanmean(scores_history[-100:]) if len(scores_history) >= 1 else np.nan
            avg_loss = episode_loss_sum / learn_steps if learn_steps > 0 else np.nan
            avg_q_val = episode_q_sum / learn_steps if learn_steps > 0 else np.nan # Avg Q of *taken* actions during learning

            log_entry = { 'episode': episode + 1, 'score': score, 'avg_score_100': avg_score,
                'steps': steps_in_episode, 'total_steps': total_steps_collected, 'epsilon': agent.epsilon,
                'avg_loss': avg_loss, 'avg_q_value': avg_q_val, 'time_elapsed_s': time.time() - start_time,
                'buffer_size': len(agent.memory) }
            training_log.append(log_entry)

            if (episode + 1) % log_interval == 0:
                avg_score_str = f"{avg_score:.2f}" if not np.isnan(avg_score) else "N/A"
                avg_loss_str = f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "N/A"
                avg_q_str = f"{avg_q_val:.3f}" if not np.isnan(avg_q_val) else "N/A"
                logger.info(f"Ep {episode+1}/{args.num_episodes} | Score: {score:.2f} | Avg(100): {avg_score_str} | Steps: {steps_in_episode} | Eps: {agent.epsilon:.3f} | AvgLoss: {avg_loss_str} | AvgQ: {avg_q_str} | StepsTotal: {total_steps_collected}") # Added total steps
                log_memory_usage() # Log memory with episode summary

            if (episode + 1) % save_interval == 0:
                agent.save_model(output_dir, model_save_prefix_template.format(episode+1))
                # Optionally save training state (scores, steps, epsilon) for better resuming
                state_data = {
                    'episode': episode + 1,
                    'total_steps': total_steps_collected,
                    'epsilon': agent.epsilon,
                    'scores_history': scores_history,
                    'episode_lengths': episode_lengths
                }
                state_save_path = output_dir / f"{model_save_prefix_template.format(episode+1)}_state.json"
                try:
                    with open(state_save_path, 'w') as f:
                        json.dump(state_data, f)
                    logger.info(f"Training state saved to {state_save_path}")
                except Exception as e:
                    logger.error(f"Failed to save training state: {e}")


    except KeyboardInterrupt: logger.warning("Training interrupted.")
    except Exception as e: logger.error(f"Training loop error: {e}", exc_info=True)
    finally:
        logger.info("Training loop finished/interrupted.")
        logger.info(f"Total time: {(time.time() - start_time)/60:.2f} min")
        try: agent.save_model(output_dir, final_model_save_prefix)
        except Exception as e: logger.error(f"Final save failed: {e}")
        if training_log:
            try: pd.DataFrame(training_log).to_csv(results_save_path, index=False); logger.info(f"Log saved: {results_save_path}")
            except Exception as e: logger.error(f"Log save failed: {e}")
        if scores_history: plot_learning_curve(scores_history, plot_save_path)
        else: logger.info("No scores, skipping plot.")
        # Clean up NVML
        if MEMORY_LOGGING_ENABLED:
            try: nvidia_smi.nvmlShutdown()
            except: pass

if __name__ == "__main__":
    main()

# --- END OF train_rl.py (v29 - Disabled CheckNumerics) ---