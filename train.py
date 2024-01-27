import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import time

env = DummyVecEnv( [ lambda: Monitor( gym.make( "airgym:airsim-drone-sample-v0",)) ])

model = PPO(
    "MultiInputPolicy",
    env,
    batch_size=128,
    learning_rate=0.001,
    verbose=1,
    max_grad_norm=10,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=100,
)
checkpoint = CheckpointCallback(100, "./checkpoints")
callbacks.append(eval_callback)
callbacks.append(checkpoint)

kwargs = {}
kwargs["callback"] = callbacks

# model.load("./checkpoints/rl_model_4400_steps")

model.learn(
    total_timesteps=5e5,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("ppo_airsim_drone_policy")
