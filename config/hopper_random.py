params = {
    'type': 'BMOPO',
    'universe': 'gym',

    'log_dir': './ray_mbpo/',  # Specify where to write log files here

    'domain': 'hopper',
    'task': 'random-v0',
    'exp_name': 'hopper_random',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 1000,
        'model_retain_epochs': 5,
        'rollout_batch_size': 50e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -3,
        'max_model_t': None,

        # For bmpo TODO
        'forward_rollout_schedule': [20, 100, 5, 5],  # [20, 100, rollout_length, rollout_length]
        'backward_rollout_schedule':[20, 100, 5, 5],  # [20, 100, rollout_length, rollout_length]
        'last_n_epoch':10,
        'backward_policy_var': 0.01,
        'n_initial_exploration_steps':5000,

        # For mopo
        'pool_load_path': 'd4rl/hopper-random-v0',
        'pool_load_max_size': 10 ** 6,
        # 'rollout_length': 5,
        'penalty_coeff': 1.0,

        'separate_mean_var': True,
        'penalty_learned_var': True,
    }
}