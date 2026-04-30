import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from model_gnn import DroneGNNModel
from env_multiagent import RayMultiDroneEnv

# 1. Đăng ký Model và Env
ModelCatalog.register_custom_model("gnn_drone_model", DroneGNNModel)

ray.init()

# 2. Cấu hình MAPPO
config = (
    PPOConfig()
    .environment(RayMultiDroneEnv, env_config={"num_drones": 3, "gui": False})
    # --- THÊM DÒNG NÀY ĐỂ TẮT API MỚI ---
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    # ------------------------------------
    .framework("torch")
    .env_runners(
        num_env_runners=1,
        num_cpus_per_env_runner=1
    )
    .learners(
        num_learners=1
    )
    .training(
        model={
            "custom_model": "gnn_drone_model",
        },
        train_batch_size=4000,
        lr=1e-4,
        lambda_=0.95,
        clip_param=0.2
    )
    .reporting(
        min_train_timesteps_per_iteration=1000, # Đẩy log lên nhanh hơn thay vì đợi đủ batch lớn
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
    )
)

# 3. Chạy huấn luyện
tuner = tune.Tuner(
    "PPO",
    run_config=tune.RunConfig(stop={"timesteps_total": 2000000}),
    param_space=config.to_dict(),
)
tuner.fit()