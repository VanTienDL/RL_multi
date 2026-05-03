import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from model_gnn import DroneGNNModel
from env_multiagent import RayMultiDroneEnv

# 1. Đăng ký model
ModelCatalog.register_custom_model("gnn_drone_model", DroneGNNModel)

# FIX: ignore_reinit_error=True để tránh lỗi nếu Ray đã được init rồi
ray.init(ignore_reinit_error=True, num_cpus=4)

# 2. Lấy obs/action space từ env để đăng ký policy đúng cách
_tmp_env = RayMultiDroneEnv({"num_drones": 3, "gui": False})
obs_space = _tmp_env.observation_space
act_space = _tmp_env.action_space
_tmp_env.close()

# 3. Cấu hình MAPPO với old API stack (tương thích Ray 2.x)
config = (
    PPOConfig()
    .environment(RayMultiDroneEnv, env_config={"num_drones": 3, "gui": False})
    # Dùng old API stack để tránh lỗi tương thích với TorchModelV2
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    .framework("torch")
    .env_runners(
        num_env_runners=1,          # 1 worker trên laptop
        num_cpus_per_env_runner=1,
    )
    # FIX: num_learners=0 → learner chạy trên main process, tiết kiệm RAM trên laptop
    .learners(num_learners=0)
    .training(
        model={"custom_model": "gnn_drone_model"},
        train_batch_size=2000,      # Giảm xuống cho laptop
        minibatch_size=256,         # FIX: dùng minibatch_size thay vì sgd_minibatch_size (Ray 2.x)
        num_epochs=5,               # FIX: dùng num_epochs thay vì num_sgd_iter (Ray 2.x)
        lr=1e-4,
        lambda_=0.95,
        clip_param=0.2,
        grad_clip=0.5,              # Thêm gradient clipping để ổn định training
    )
    .reporting(
        min_train_timesteps_per_iteration=500,
    )
    # FIX: policies phải là dict, không phải set
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

# 4. Chạy training
tuner = tune.Tuner(
    "PPO",
    run_config=tune.RunConfig(
        stop={"timesteps_total": 500_000},  # Giảm target để test trước
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
    ),
    param_space=config.to_dict(),
)
tuner.fit()