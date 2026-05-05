"""
train.py — Script training cho cả Low-level DQN và High-level PPO/GNN.

Dùng:
  python train.py --level low    # Train DQN trước
  python train.py --level high   # Train GNN PPO sau khi có DQN tốt
  python train.py --level both   # Train tuần tự (low → high)
"""

import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ══════════════════════════════════════════════════════════════
# LOW-LEVEL: DQN với Stable-Baselines3 (đơn giản hơn Ray)
# ══════════════════════════════════════════════════════════════

def train_lowlevel(total_steps: int = 200_000):
    """
    Train DQN cho low-level navigation.
    Dùng SB3 thay vì Ray vì:
      - Single agent → không cần multi-agent infra
      - DQN trong SB3 mature hơn, ít bug hơn
      - Nhanh hơn nhiều trên CPU laptop
    """
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from env_lowlevel import LowLevelStandaloneDQN

    print("=" * 60)
    print("  TRAINING LOW-LEVEL DQN")
    print("=" * 60)

    # Tạo env (vectorized để tận dụng đa luồng nếu có)
    env = make_vec_env(
        LowLevelStandaloneDQN,
        n_envs=2,  # 2 env song song trên CPU
        env_kwargs={"config": {"num_drones": 3, "gui": False}},
    )

    eval_env = LowLevelStandaloneDQN(config={"num_drones": 3, "gui": False})

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,        # Giảm xuống để vừa RAM laptop
        learning_starts=1_000,
        batch_size=64,
        tau=0.005,                 # Soft update target network
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,  # Khám phá nhiều ở đầu
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/lowlevel/",
        policy_kwargs=dict(
            net_arch=[128, 128],   # 2 hidden layers
        ),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="./models/lowlevel/",
            log_path="./logs/lowlevel/eval/",
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=20_000,
            save_path="./models/lowlevel/checkpoints/",
            name_prefix="dqn_lowlevel",
        ),
    ]

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save("./models/lowlevel/dqn_final")
    print("\n✓ Low-level DQN saved to ./models/lowlevel/dqn_final")
    return model


# ══════════════════════════════════════════════════════════════
# HIGH-LEVEL: PPO + GNN với Ray RLlib
# ══════════════════════════════════════════════════════════════

def train_highlevel(total_steps: int = 500_000):
    """
    Train high-level GNN PPO với Ray RLlib (MAPPO).
    """
    import ray
    from ray import tune
    from ray.rllib.models import ModelCatalog
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    from model_gnn import HighLevelGNNModel
    from env_highlevel import HighLevelMAEnv

    print("=" * 60)
    print("  TRAINING HIGH-LEVEL GNN PPO (MAPPO)")
    print("=" * 60)

    ModelCatalog.register_custom_model("gnn_highlevel", HighLevelGNNModel)
    ray.init(ignore_reinit_error=True, num_cpus=4)

    # Lấy obs/action space
    _tmp = HighLevelMAEnv({"num_drones": 3, "gui": False})
    obs_space = _tmp.observation_space
    act_space = _tmp.action_space
    _tmp.close()

    config = (
        PPOConfig()
        .environment(HighLevelMAEnv, env_config={"num_drones": 3, "gui": False})
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
        .learners(num_learners=0)   # Main process = tiết kiệm RAM
        .training(
            model={"custom_model": "gnn_highlevel"},
            train_batch_size=2000,
            minibatch_size=256,
            num_epochs=5,
            lr=3e-4,
            lambda_=0.95,
            clip_param=0.2,
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,    # Khuyến khích khám phá
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                )
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            stop={"timesteps_total": total_steps},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=20,
                checkpoint_at_end=True,
            ),
        ),
        param_space=config.to_dict(),
    )

    results = tuner.fit()
    print("\n✓ High-level GNN PPO training complete.")
    return results


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=["low", "high", "both"],
        default="low",
        help="Chọn level cần train",
    )
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("./models/lowlevel/checkpoints", exist_ok=True)
    os.makedirs("./logs/lowlevel/eval",          exist_ok=True)

    if args.level == "low":
        train_lowlevel(args.steps or 200_000)
    elif args.level == "high":
        train_highlevel(args.steps or 500_000)
    elif args.level == "both":
        print("Bước 1: Train Low-level DQN")
        train_lowlevel(args.steps or 200_000)
        print("\nBước 2: Train High-level GNN")
        train_highlevel(args.steps or 500_000)