from copy import deepcopy


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_BMPO_algorithm(variant, *args, **kwargs):
    import bmpo

    algorithm = bmpo.BMPO(*args, **kwargs)

    return algorithm

# Create BMOPO_algorithm
def create_BMOPO_algorithm(variant, *args, **kwargs):
    # TODO: implement the BMOPO algo
    from bmopo import BMOPO

    algorithm = BMOPO(*args, **kwargs)

    return algorithm

ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'BMPO': create_BMPO_algorithm,
    'BMOPO': create_BMOPO_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    """ An example:
    algorithm_kwargs = {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': True,

        - 'discount': 0.99,
        - 'tau': 5e-3,
        - 'reward_scale': 1.0,

        - 'model_train_freq': 1000,
        - 'model_retain_epochs': 5,
        - 'rollout_batch_size': 50e3,
        - 'deterministic': False,
        - 'num_networks': 7,
        - 'num_elites': 5,
        - 'real_ratio': 0.05,
        - 'target_entropy': -3,
        - 'max_model_t': None,

        # For bmpo TODO
        - 'forward_rollout_schedule': [20, 100, 1, 1],
        - 'backward_rollout_schedule':[20, 100, 1, 1],
        - 'last_n_epoch':10,
        - 'planning_horizon':0,
        - 'backward_policy_var': 0.01,
        'n_initial_exploration_steps':5000,

        # For mopo
        - 'pool_load_path': 'd4rl/halfcheetah-medium-v0',
        - 'pool_load_max_size': 10 ** 6,
        - 'rollout_length': 1,  # TODO: forward_length, backward_length
        - 'penalty_coeff': 1.0,

        - 'separate_mean_var': True,
        - 'penalty_learned_var': True,

        ---> from main.py
        - 'n_epochs': NUM_EPOCHS_PER_DOMAIN[domain]  # TODO: mopo seems doesn't use the "n_epochs", check it

        - 'reparameterize': True
        - 'lr': 3e-4
        - 'target_update_interval': 1
        - 'tau': 5e-3
        'store_extra_policy_info': False
        'action_prior': 'uniform'
    }

    kwargs =  {variant=self.variant,
            - training_environment=training_environment,
            - evaluation_environment=evaluation_environment,
            - policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            - Qs=Qs,
            - pool=replay_pool,
            - static_fns=static_fns,
            - sampler=sampler,
            - session=self._session,
            - log_file='./log/%s/%d.log' % (self.variant['algorithm_params']['domain'], time.time()))
            }
    """
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
