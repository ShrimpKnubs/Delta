# --- START OF src/d3qn_agent.py (v6 - Fixed Custom Layer Import for WSL) ---

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Optimizer, Adam, AdamW
from collections import deque
import random
import logging
import os
from pathlib import Path
from typing import Union, Tuple
import gc
import sys # Import sys to check path if needed

logger = logging.getLogger(__name__) # Get logger instance

# --- Custom Layer Import Handling --- MODIFIED
try:
    # Assume train_rl.py added project root to sys.path
    # Directly import from the top-level module name
    from train_model import PositionalEmbedding, TransformerBlock
    custom_layers_available = True
    # Register custom objects for saving/loading
    tf.keras.utils.register_keras_serializable(package="Custom", name="PositionalEmbedding")(PositionalEmbedding)
    tf.keras.utils.register_keras_serializable(package="Custom", name="TransformerBlock")(TransformerBlock)
    logger.info("Successfully imported and registered custom layers from train_model.")
except ImportError as e:
    logger.error(f"Could not import custom layers from train_model: {e}", exc_info=True)
    logger.error(f"Ensure 'train_model.py' is in the project root added to sys.path: {sys.path}")
    # Define dummy classes if import fails
    custom_layers_available = False
    logging.warning("Using dummy layers for PositionalEmbedding/TransformerBlock.")
    class PositionalEmbedding(layers.Layer): # Dummy
        def __init__(self, sequence_length, embed_dim, **kwargs): super().__init__(**kwargs); self.sl=sequence_length; self.ed=embed_dim
        def call(self, inputs): return inputs
        def get_config(self): config = super().get_config(); config.update({"sequence_length": self.sl, "embed_dim": self.ed}); return config
    class TransformerBlock(layers.Layer): # Dummy
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs): super().__init__(**kwargs); self.ed=embed_dim; self.nh=num_heads; self.ffd=ff_dim; self.r=rate
        def call(self, inputs, training=False, mask=None): return inputs
        def get_config(self): config = super().get_config(); config.update({"embed_dim": self.ed, "num_heads": self.nh, "ff_dim": self.ffd, "rate": self.r}); return config
# --- End Custom Layer Import ---


class D3QNAgent:
    """ Dueling Double Deep Q-Network Agent (v6) """

    # --- __init__ --- (No changes needed inside __init__ itself from v5)
    def __init__(self,
                 state_shape: Tuple[int, int],
                 action_space_n: int,
                 optimizer: Optimizer,
                 model_params_config: dict,
                 rl_params_config: dict,
                 ):
        self.state_shape = state_shape
        self.action_space_n = action_space_n
        self.optimizer = optimizer
        self.arch_params = model_params_config.get('model_params', {})
        self.rl_params = rl_params_config
        self.gamma = self.rl_params.get('gamma', 0.99)
        self.epsilon = self.rl_params.get('epsilon_start', 1.0)
        self.epsilon_min = self.rl_params.get('epsilon_end', 0.05)
        self.epsilon_decay_steps = self.rl_params.get('epsilon_decay_steps', 100000)
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps if self.epsilon_decay_steps > 0 else 0
        self.batch_size = self.rl_params.get('batch_size', 64)
        self.buffer_size = self.rl_params.get('buffer_size', 50000)
        self.target_update_freq = self.rl_params.get('target_update_freq', 1000)
        self.learning_starts = self.rl_params.get('learning_starts', 1000)
        self.memory = deque(maxlen=self.buffer_size)
        self.projection_layer = None # Ensure projection layer is reset/defined in build
        self.q_network = self._build_d3qn_model()
        self.target_q_network = self._build_d3qn_model()
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.q_network.compile(optimizer=self.optimizer, loss=tf.keras.losses.Huber())
        self.train_step_counter = 0
        logger.info("D3QN Agent Initialized.")
        logger.info(f" State Shape: {self.state_shape}, Actions: {self.action_space_n}")
        self.q_network.summary(print_fn=logger.info) # Log summary after compile

    # --- _build_d3qn_model --- (No changes needed here from v5/v4)
    def _build_d3qn_model(self) -> Model:
        input_shape = self.state_shape
        embed_dim = self.arch_params.get('embed_dim', 64)
        conv_filters = self.arch_params.get('conv_filters', [24, 24, 24])
        conv_kernels = self.arch_params.get('conv_kernels', [3, 7, 15])
        conv_dropout = self.arch_params.get('conv_dropout', 0.25)
        pool_size = self.arch_params.get('pool_size', 2)
        num_heads = self.arch_params.get('num_heads', 4)
        ff_dim = self.arch_params.get('ff_dim', 128)
        num_transformer_blocks = self.arch_params.get('num_transformer_blocks', 1)
        transformer_dropout = self.arch_params.get('transformer_dropout', 0.25)
        dense_units = self.arch_params.get('dense_units', [64])
        dense_dropout = self.arch_params.get('dense_dropout', 0.3)
        inputs = layers.Input(shape=input_shape, name='input_features')
        x = inputs
        last_conv_features = input_shape[-1]
        if conv_filters and conv_kernels and len(conv_filters) == len(conv_kernels):
            conv_outputs = [layers.Conv1D(filters=f, kernel_size=k, padding='causal', activation='relu', name=f'conv1d_{i+1}_k{k}')(x)
                            for i, (f, k) in enumerate(zip(conv_filters, conv_kernels))]
            if len(conv_outputs) > 1: x = layers.Concatenate(name='concat_conv')(conv_outputs); last_conv_features = sum(conv_filters)
            elif len(conv_outputs) == 1: x = conv_outputs[0]; last_conv_features = conv_filters[0]
        else: logger.info("No conv layers.")
        pooled_len = input_shape[0]
        if pool_size > 1 and input_shape[0] is not None and input_shape[0] >= pool_size:
            x = layers.MaxPooling1D(pool_size=pool_size, name='max_pooling')(x); pooled_len = input_shape[0] // pool_size
        else: logger.info("No pooling.")
        current_features = last_conv_features
        # Use Python 'if' for build-time conditional layer creation
        if current_features != embed_dim:
            logger.info(f"Applying projection layer: {current_features} -> {embed_dim}")
            # Ensure self.projection_layer is defined if needed, re-use if exists
            if self.projection_layer is None: self.projection_layer = layers.Dense(embed_dim, name='feature_projection')
            x = self.projection_layer(x)
        else: logger.info("Skipping projection.")
        x = layers.LayerNormalization(epsilon=1e-6, name='post_conv_norm')(x)
        x = layers.Dropout(conv_dropout, name='post_conv_dropout')(x)
        pos_embed_seq_len = -1
        if pooled_len is not None and pooled_len > 0:
            pos_embed_seq_len = pooled_len
            # Use the imported (or dummy) PositionalEmbedding class
            try: x = PositionalEmbedding(pos_embed_seq_len, embed_dim, name='pos_embedding')(x)
            except Exception as e: logger.error(f"PosEmb Error (len={pos_embed_seq_len}, dim={embed_dim}): {e}"); raise e
        else: logger.warning("Skipping PositionalEmbedding.")
        # Use the imported (or dummy) TransformerBlock class
        for i in range(num_transformer_blocks): x = TransformerBlock(embed_dim, num_heads, ff_dim, transformer_dropout, name=f'transformer_{i+1}')(x)
        features = layers.GlobalAveragePooling1D(name='gap')(x); shared = features
        for i, units in enumerate(dense_units): shared = layers.Dense(units, activation='relu', name=f'shared_dense_{i+1}')(shared); shared = layers.Dropout(dense_dropout, name=f'shared_dropout_{i+1}')(shared)
        value_stream = layers.Dense(1, name='value_output')(shared); advantage_stream = layers.Dense(self.action_space_n, name='advantage_raw')(shared)
        def combine_streams(streams): v, a = streams; mean_advantage = tf.reduce_mean(a, axis=1, keepdims=True); q_values = v + (a - mean_advantage); return q_values
        q_values = layers.Lambda(combine_streams, name='q_values_output')([value_stream, advantage_stream])
        model = Model(inputs=inputs, outputs=q_values, name='D3QN_Model'); return model

    # --- store_transition, choose_action, _update_epsilon, _update_target_network remain the same ---
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, float(done)))

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_space_n)
        if state.ndim == len(self.state_shape): state = np.expand_dims(state, axis=0)
        q_values = self.q_network(tf.convert_to_tensor(state, dtype=tf.float32), training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon -= self.epsilon_delta; self.epsilon = max(self.epsilon_min, self.epsilon)

    def _update_target_network(self):
        logger.info(f"--- Updating target network at train step {self.train_step_counter} ---")
        self.target_q_network.set_weights(self.q_network.get_weights())

    # --- learn method remains the same as v5 ---
    def learn(self) -> tuple[float, float]:
        if len(self.memory) < self.learning_starts: return np.nan, np.nan
        if len(self.memory) < self.batch_size: return np.nan, np.nan
        minibatch = random.sample(self.memory, self.batch_size)
        states = tf.stack([tf.convert_to_tensor(item[0], dtype=tf.float32) for item in minibatch])
        actions = tf.stack([tf.convert_to_tensor(item[1], dtype=tf.int32) for item in minibatch])
        rewards = tf.stack([tf.convert_to_tensor(item[2], dtype=tf.float32) for item in minibatch])
        next_states = tf.stack([tf.convert_to_tensor(item[3], dtype=tf.float32) for item in minibatch])
        dones = tf.stack([tf.convert_to_tensor(item[4], dtype=tf.float32) for item in minibatch])
        q_next_main = self.q_network(next_states, training=False)
        best_actions_next = tf.argmax(q_next_main, axis=1)
        q_next_target_all = self.target_q_network(next_states, training=False)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, tf.cast(best_actions_next, tf.int32)], axis=1)
        q_val_next = tf.gather_nd(q_next_target_all, gather_indices)
        td_target = rewards + self.gamma * q_val_next * (1.0 - dones)
        loss_value = tf.constant(np.nan, dtype=tf.float32)
        avg_q_taken = tf.constant(np.nan, dtype=tf.float32)
        gradients = None
        try:
            with tf.GradientTape() as tape:
                q_current_all = self.q_network(states, training=True)
                gather_indices_current = tf.stack([batch_indices, actions], axis=1)
                q_current_taken = tf.gather_nd(q_current_all, gather_indices_current)
                loss_value = self.q_network.compiled_loss(td_target, q_current_taken, regularization_losses=self.q_network.losses)
            if not tf.math.is_nan(loss_value) and not tf.math.is_inf(loss_value): gradients = tape.gradient(loss_value, self.q_network.trainable_variables)
            else: logger.warning(f"Skipping grad calc: Loss={loss_value.numpy()}")
        except Exception as e: logger.error(f"Error during loss/Tape: {e}", exc_info=True)
        if gradients is not None:
            valid_gradients = True
            for i, grad in enumerate(gradients):
                 if grad is None: logger.warning(f"Grad None: {self.q_network.trainable_variables[i].name}"); valid_gradients = False; break
                 if tf.reduce_any(tf.math.is_nan(grad)): logger.warning(f"NaN in grad: {self.q_network.trainable_variables[i].name}"); valid_gradients = False; break
            if valid_gradients:
                try: self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables)); avg_q_taken = tf.reduce_mean(q_current_taken)
                except Exception as e_apply: logger.error(f"Error applying grads: {e_apply}", exc_info=True); loss_value=tf.constant(np.nan); avg_q_taken=tf.constant(np.nan)
            else: logger.warning("Skipping step: Invalid grads."); loss_value=tf.constant(np.nan); avg_q_taken=tf.constant(np.nan)
        else: avg_q_taken=tf.constant(np.nan)
        self.train_step_counter += 1
        self._update_epsilon()
        if self.train_step_counter % self.target_update_freq == 0: self._update_target_network()
        if self.train_step_counter % 100 == 0: gc.collect() # Garbage collect less frequently
        loss_py = float(loss_value.numpy()) if not tf.math.is_nan(loss_value) else np.nan
        avg_q_py = float(avg_q_taken.numpy()) if not tf.math.is_nan(avg_q_taken) else np.nan
        return loss_py, avg_q_py

    # --- save_model, load_model remain the same ---
    def save_model(self, directory: Union[str, Path], filename_prefix: str):
        dir_path = Path(directory); dir_path.mkdir(parents=True, exist_ok=True)
        main_path = dir_path / f"{filename_prefix}_q_network.weights.h5"
        target_path = dir_path / f"{filename_prefix}_target_q_network.weights.h5"
        try:
            if not self.q_network.built: self.q_network.build((None,) + self.state_shape)
            if not self.target_q_network.built: self.target_q_network.build((None,) + self.state_shape)
            self.q_network.save_weights(str(main_path))
            self.target_q_network.save_weights(str(target_path))
            logger.info(f"Agent models saved: {main_path}, {target_path}")
        except Exception as e: logger.error(f"Error saving models: {e}", exc_info=True)

    def load_model(self, directory: Union[str, Path], filename_prefix: str):
        dir_path = Path(directory)
        main_path = dir_path / f"{filename_prefix}_q_network.weights.h5"
        target_path = dir_path / f"{filename_prefix}_target_q_network.weights.h5"
        try:
            if not main_path.is_file() or not target_path.is_file(): logger.error(f"Weights not found: {main_path} or {target_path}"); return False
            if not self.q_network.built: self.q_network.build((None,) + self.state_shape)
            if not self.target_q_network.built: self.target_q_network.build((None,) + self.state_shape)
            self.q_network.load_weights(str(main_path))
            self.target_q_network.load_weights(str(target_path))
            logger.info(f"Agent models loaded from prefix: {filename_prefix} in {dir_path}")
            return True
        except Exception as e: logger.error(f"Error loading models: {e}", exc_info=True); return False

# --- END OF src/d3qn_agent.py (v6 - Fixed Custom Layer Import for WSL) ---