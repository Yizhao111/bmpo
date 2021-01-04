import argparse
import importlib
import runner
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.halfcheetah_mixed')
# parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

module = importlib.import_module(args.config)
params = getattr(module, 'params')  # get config dict params from --config=config.xx.py

universe, domain, task = params['universe'], params['domain'], params['task']
epoch_length = params['kwargs']['epoch_length']

variant_spec = {
        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
                'squash': True,
            }
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
            }
        },

        'algorithm_params': params,

        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': epoch_length,
                'min_pool_size': epoch_length,
                'batch_size': 256,
            }
        },
        'run_params': {
            # 'seed': args.seed,  # TODO: SEED?
            'checkpoint_at_end': True,
            'checkpoint_frequency': 20,
            'checkpoint_replay_pool': False,
        },
    }

print('------------------ Experimental Parameters --------------')
for k, v in variant_spec.items():
    print(' ' * 5 + k + ': ')
    for sub_k, sub_v in v.items():
        if sub_k == 'kwargs':
            print(' ' * 10 + sub_k + ': ')
            for kwargs_k, kwargs_v in sub_v.items():
                print(' '* 20 + kwargs_k + ': '+ str(kwargs_v))
        else:
            print(' '* 10 + sub_k + ': ' + str(sub_v))
if __name__ == "__main__":
    exp_runner = runner.ExperimentRunner(variant_spec)
    diagnostics = exp_runner.train()  # train agent and return diagnostics
