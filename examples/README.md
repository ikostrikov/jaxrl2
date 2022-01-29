
# Online RL

## States

### SAC
```bash
python train_online.py --env_name=HalfCheetah-v2
```

## Pixels

### DrQ
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online_pixels.py --env_name=cheetah-run-v0
```

# Offline RL

## States

### BC
```bash
python train_offline.py --config=configs/offline_config.py:bc --config.model_config.distr=unitstd_normal --env_name=halfcheetah-expert-v2
```

### %BC
```bash
python train_offline.py --config=configs/offline_config.py:bc --config.model_config.distr=unitstd_normal --env_name=halfcheetah-medium-expert-v2 --filter_percentile=10
```

### fBC (filtered BC)
```bash
python train_offline.py --config=configs/offline_config.py:bc --config.model_config.distr=unitstd_normal --env_name=antmaze-large-play-v2 --filter_threshold=0.5
```

### BC (Autoregressive Policy)
```bash
python train_offline.py --config=configs/offline_config.py:bc --config.model_config.distr=ar --env_name=halfcheetah-expert-v2
```

### IQL
#### AntMaze
```bash
python train_offline.py --config=configs/offline_config.py:iql_antmaze --env_name=antmaze-large-play-v2 --eval_interval=100000 --eval_episodes=100
```
#### Locomotion
```bash
python train_offline.py --config=configs/offline_config.py:iql_mujoco --env_name=halfcheetah-medium-expert-v2 --eval_interval=100000 --eval_episodes=100
```

## Pixels

### Collect data
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online_pixels.py --env_name=cheetah-run-v0 --save_buffer
```

### PixelBC
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline_pixels.py --env_name=cheetah-run-v0 --config=configs/offline_pixels_config.py:bc
```

### PixelIQL
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline_pixels.py --env_name=cheetah-run-v0 --config=configs/offline_pixels_config.py:iql
```