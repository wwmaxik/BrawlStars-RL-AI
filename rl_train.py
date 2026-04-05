import os
import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import BrawlStarsYoloEnv
from dashboard import DashboardData, start_dashboard
from torch.utils.tensorboard import SummaryWriter
from constants import Constants
import torch

class TrainStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
        self.best_model_path = "brawl_yolo_recurrent_ppo_best.zip"
        
    def _init_callback(self) -> None:
        # Инициализация TensorBoard
        log_dir = "./logs/tensorboard"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def _on_step(self) -> bool:
        # Update dashboard stats from logger
        logs = self.model.logger.name_to_value
        
        DashboardData.train_stats.update({
            "Learning Rate": logs.get("train/learning_rate", 0),
            "Entropy Loss": logs.get("train/entropy_loss", 0),
            "Value Loss": logs.get("train/value_loss", 0),
            "Clip Fraction": logs.get("train/clip_fraction", 0),
            "Mean Reward": logs.get("rollout/ep_rew_mean", 0),
            "Mean Len": logs.get("rollout/ep_len_mean", 0),
            "Explained Var": logs.get("train/explained_variance", 0)
        })
        
        # Логирование в TensorBoard
        for key, value in logs.items():
            self.writer.add_scalar(key, value, self.num_timesteps)
        
        # Отслеживание лучшей модели
        mean_reward = logs.get("rollout/ep_rew_mean", 0)
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.writer.add_scalar("best/model_reward", mean_reward, self.num_timesteps)
            print(f"\n[BEST MODEL] New best mean reward: {mean_reward:.2f} - Saving...")
            self.model.save(self.best_model_path.replace(".zip", ""))
        
        return True
    
    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()
            print("\n[TensorBoard] Logs saved to ./logs/tensorboard")

def main():
    print("=== Brawl Stars Bot RL Training (Liquid AI / RecurrentPPO v2.0) ===")
    print("=== IMPROVED: Extended observation, Frame stacking, MultiDiscrete actions ===")
    print("Warning: RL training via PyAutoGUI and BlueStacks in real-time is extremely slow.")
    print("Make sure Brawl Stars is open, in the lobby or loading screen, and solo showdown is selected.")
    print("Press Ctrl+C in this console to stop and save the model.")
    time.sleep(3)

    # Start the web dashboard
    start_dashboard()

    # Создаём environment
    env = BrawlStarsYoloEnv(frame_stack=4)

    model_path = "brawl_yolo_recurrent_ppo.zip"
    best_model_path = "brawl_yolo_recurrent_ppo_best.zip"

    # IMPROVED Hyperparameters
    policy_kwargs = dict(
        net_arch=[
            dict(vision=[128, 128], pi=[256, 128], vf=[256, 128])
        ],
        lstm_hidden_size=256,
    )

    # Check if we have a saved model to continue training
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = RecurrentPPO.load(
            model_path, 
            env=env, 
            verbose=1, 
            device="cuda",
            tensorboard_log="./logs/tensorboard",
        )
    else:
        print("No existing model found. Initializing a new one with IMPROVED hyperparameters...")
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            verbose=1,
            device="cuda",
            # Увеличено для лучшей стабильности с LSTM
            n_steps=4096,
            batch_size=128,
            learning_rate=1e-4,
            gamma=0.995,
            ent_coef=0.05,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            n_epochs=10,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/tensorboard",
        )

    # Configure callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path="./logs/", 
        name_prefix="brawl_ai_checkpoint"
    )
    stats_callback = TrainStatsCallback()
    callback_list = CallbackList([checkpoint_callback, stats_callback])

    try:
        print("Starting training with IMPROVED configuration...")
        print(f"  - n_steps: 4096")
        print(f"  - batch_size: 128")
        print(f"  - learning_rate: 1e-4")
        print(f"  - gamma: 0.995")
        print(f"  - ent_coef: 0.05")
        print(f"  - lstm_hidden_size: 256")
        print(f"  - Frame stacking: 4")
        print(f"  - Action space: MultiDiscrete([3,3,2,2,2])")
        print(f"  - Observation vector: {Constants.vector_size} dimensions")
        
        model.learn(
            total_timesteps=2000000,  # Увеличено с 1M до 2M
            callback=callback_list, 
            reset_num_timesteps=False,
            tb_log_name="RecurrentPPO_v2",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Сохраняем финальную модель
        model.save("brawl_yolo_recurrent_ppo")
        print("Final model saved to 'brawl_yolo_recurrent_ppo.zip'.")
        
        if os.path.exists(best_model_path):
            print(f"Best model saved to '{best_model_path}'.")
        
        env.close()

if __name__ == "__main__":
    main()
