import math
from collections import OrderedDict
from numbers import Number
from itertools import count
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util
import wandb

import scipy.stats as stats

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from models.constructor import construct_forward_model, format_samples_for_forward_training, construct_backward_model, format_samples_for_backward_training
from models.fake_env import Forward_FakeEnv, Backward_FakeEnv

from utils.loader import restore_pool # load d4rl dataset
from utils.filesystem import mkdir

def td_target(reward, discount, next_value):
    return reward + discount * next_value


class BMOPO(RLAlgorithm):

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,  # used to store env samples
            static_fns,
            log_file=None,
            plotter=None, # not used, can be used to draw Q function
            tf_summaries=False, # not used here

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,  # rewards discount
            tau=5e-3,  # ratio when updating the target_Q function.
            target_update_interval=1,  # Frequency at which target network updates occur in iterations.
            action_prior='uniform',
            reparameterize=False,  # If True, we use a gradient estimator for the policy derived using the reparameterization trick. We use a likelihood ratio based estimator otherwise.
            store_extra_policy_info=False,

            deterministic=False,  # whether to use deterministic model (mean) when unrolling model to collect samples
            model_train_freq=250,  # Frequency to (train) unroll the model
            num_networks=7,  # The amount of ensembles
            num_elites=5,  # selected amount of elite ensembles
            model_retain_epochs=20,
            rollout_batch_size=100e3, # The batch size when unrolling the model, the total amount of collected samples is batch * length
            real_ratio=0.1,  # The ratio of env_data/model_data when feeding training data

            forward_rollout_schedule=None,  # forward rollout schedule: [min_epoch, max_epoch, min_length, max_length]
            backward_rollout_schedule=None,

            last_n_epoch=10,  # TODO: HOW TO USE IT? last n epoch data to collect used for training the backward policy
            backward_policy_var=0,  # A fixed var for backward policy
            hidden_dim=200,  # hidden_dim of the nn model
            max_model_t=None,  # the maximal training time

            # TODO: not existed in bmpo but in mopo
            pool_load_path='', # the path to load d4rl dataset
            pool_load_max_size=0,  # the max size of load data
            # The penalty term used for offline setting
            separate_mean_var=False, #TODO: Use this latter
            penalty_coeff=0.,
            penalty_learned_var=False,

            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(BMOPO, self).__init__(**kwargs)

        obs_dim = np.prod(training_environment.observation_space.shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._forward_model = construct_forward_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites)
        self._backward_model = construct_backward_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                         num_networks=num_networks, num_elites=num_elites)
        self._static_fns = static_fns
        self.f_fake_env = Forward_FakeEnv(self._forward_model, self._static_fns)
        self.b_fake_env = Backward_FakeEnv(self._backward_model, self._static_fns)

        self._forward_rollout_schedule = forward_rollout_schedule
        self._backward_rollout_schedule = backward_rollout_schedule
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._log_dir = os.getcwd()

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._last_n_epoch = int(last_n_epoch)
        self._backward_policy_var = backward_policy_var

        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape
        self.log_file = log_file

        self._build()

        ## Load replay pool data (load d4rl dataset) TODO: add pool_load_path, pool_load_max_size into params
        self._pool_load_path = pool_load_path
        self._pool_load_max_size = pool_load_max_size
        # load samples from d4rl
        restore_pool(self._pool, self._pool_load_path, self._pool_load_max_size, save_path=self._log_dir)
        self._init_pool_size = self._pool.size
        print('[ BMOPO ] Staring with pool size: {}'.format(self._init_pool_size))

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._build_backward_policy(self._act_dim)

    def _build_backward_policy(self,act_dim):
        self._max_logvar = tf.Variable(np.ones([1, act_dim]), dtype=tf.float32,
                                       name="max_log_var")
        self._min_logvar = tf.Variable(-np.ones([1, act_dim]) * 10., dtype=tf.float32,
                                       name="min_log_var")
        self._before_action_mean, self._before_action_logvar = self._backward_policy_net('backward_policy',
                                                                                       self._next_observations_ph,
                                                                                       act_dim)
        action_logvar = self._max_logvar - tf.nn.softplus(self._max_logvar - self._before_action_logvar)
        action_logvar = self._min_logvar + tf.nn.softplus(action_logvar - self._min_logvar)
        self._before_action_var = tf.exp(action_logvar)
        self._backward_policy_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='backward_policy')
        loss1 = tf.reduce_mean(tf.square(self._before_action_mean - self._actions_ph) / self._before_action_var)
        loss2 = tf.reduce_mean(tf.log(self._before_action_var))
        self._backward_policy_loss = loss1 + loss2
        self._backward_policy_optimizer = tf.train.AdamOptimizer(self._policy_lr).minimize(loss=self._backward_policy_loss,
                                                                                 var_list=self._backward_policy_params)

    def _backward_policy_net(self, scope, state,action_dim,hidden_dim = 256):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden_layer1 = tf.layers.dense(state, hidden_dim, tf.nn.relu)
            hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_dim, tf.nn.relu)
            return tf.tanh(tf.layers.dense(hidden_layer2, action_dim)), \
                   tf.layers.dense(hidden_layer2, action_dim)

    def _get_before_action(self,obs):
        # the backward policy
        before_action_mean, before_action_var = self._session.run(
            [self._before_action_mean, self._before_action_var],
            feed_dict={
                self._next_observations_ph: obs
            })
        if(self._backward_policy_var != 0):
            before_action_var = self._backward_policy_var
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(before_action_mean),
                            scale=np.ones_like(before_action_mean))
        before_actions = X.rvs(size=np.shape(before_action_mean)) * np.sqrt(
            before_action_var) + before_action_mean  # sample from backward policy
        act = np.clip(before_actions, -1, 1)
        return act

    def _train(self):

        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool  # init the pool for env data (to store d4rl in offline setting)
        f_model_metrics, b_model_metrics = {},{}  # to store the model metrics, for logging

        if not self._training_started:
            self._init_training()

        self.sampler.initialize(training_environment, policy, pool)

        self._training_before_hook()

        # max_epochs = 1 if (self._forward_model.model_loaded and self._backward_model.model_loaded) else None
        max_epochs = 2 # TODO: CHAGE IT
        # train the forward and the backward dynamic model
        f_model_train_metrics, b_model_train_metrics = self._train_model(batch_size=256,
                                                                        max_epochs=max_epochs,
                                                                        holdout_ratio=0.2,
                                                                        max_t=self._max_model_t)
        f_model_metrics.update(f_model_train_metrics)
        b_model_metrics.update(b_model_train_metrics)

        self._log_model()  # print and save model

        # collect samples via unrolling models and use them to update the policy
        for self._epoch in range(self._epoch, self._n_epochs):  # self._n_epochs = n_epochs, and it's initialised in RLAlgo

            self._epoch_before_hook()

            start_samples = self.sampler._total_samples
            print('------- epoch: {} --------'.format(self._epoch), "start_samples", start_samples)
            print('[ True Env Buffer Size ]', pool.size)

            for timestep in count():
                # samples_now = self.sampler._total_samples
                # self._timestep = samples_now - start_samples
                self._timestep = timestep

                # if samples_now >= start_samples + self._epoch_length and self.ready_to_train:
                if timestep >= self._epoch_length:
                    break

                self._timestep_before_hook()

                # Rollout the dynamic model to collect more data every self._model_train_freq (1000)
                if self._timestep % self._model_train_freq == 0:

                    self._set_rollout_length()  # set rollout_length for backward and forward model according to self._b/f_rollout_schedule
                    self._reallocate_model_pool()  # init the data pool used to store samples via unrolling the dynamic model
                    f_model_rollout_metrics, b_model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size,
                                        deterministic=self._deterministic)
                    f_model_metrics.update(f_model_rollout_metrics)
                    b_model_metrics.update(b_model_rollout_metrics)
                # Train the actor and the critic (forward and backward).
                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)  # Here, for bmopo, need to train both forward and backward policy.

                self._timestep_after_hook()

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))  # TODO: this seems wrong: param==1, chech the mopo code
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
            else:
                evaluation_metrics = {}

            self._epoch_after_hook(training_paths)

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    (f'forward-model/{key}', f_model_metrics[key])
                    for key in sorted(f_model_metrics.keys())
                ),
                *(
                    (f'backward-model/{key}', b_model_metrics[key])
                    for key in sorted(b_model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            # logging via wandb
            wandb.log(diagnostics)

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)
            print(diagnostics)
            f_log = open(self.log_file, 'a')
            f_log.write('epoch: %d\n' % self._epoch)
            f_log.write('total time steps: %d\n' % self._total_timestep)
            f_log.write('evaluation return: %f\n' % evaluation_metrics['return-average'])
            f_log.close()

        self.sampler.terminate()

        self._training_after_hook()

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _log_policy(self):
        print('--------- log policy is passed ---------')
        pass

    def _log_model(self):
        if self._forward_model.model_loaded and self._backward_model.model_loaded:
            print('[ MOPO ] Loaded model, skipping save')
        else:
            save_path = os.path.join(self._log_dir, 'models')
            mkdir(save_path)
            print('[ MOPO ] Saving model to: {}'.format(save_path))

            # TODO: check the save function
            self._forward_model.save(save_path, self._total_timestep)
            self._backward_model.save(save_path, self._total_timestep)
    def _set_rollout_length(self):
        # set the rollout_length according to self._backward_rollout_schedule and self._forward_rollout_schedule.
        # the format of the rollout_schedule is [min_epoch, max_epoch, min_length, max_length]

        #set backward rollout length
        min_epoch, max_epoch, min_length, max_length = self._backward_rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._backward_rollout_length = int(y)
        print('[ Set Backward Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._backward_rollout_length, min_length, max_length
        ))
        # set forward rollout length
        min_epoch, max_epoch, min_length, max_length = self._forward_rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._forward_rollout_length = int(y)
        print('[ Set Forward Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._forward_rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        # init the data pool used to store samples from unrolling the dynamic model
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int((self._forward_rollout_length + self._backward_rollout_length) * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ Allocate Model Pool ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ Reallocate Model Pool ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        # get all the env samples and use them to train the backward and forward dynamic model
        env_samples = self._pool.return_all_samples()
        print('Training forward model:')
        train_inputs, train_outputs = format_samples_for_forward_training(env_samples)
        f_model_metrics = self._forward_model.train(train_inputs, train_outputs, **kwargs)
        print('Training backward model:')
        train_inputs, train_outputs = format_samples_for_backward_training(env_samples)
        b_model_metrics = self._backward_model.train(train_inputs, train_outputs, **kwargs)
        return f_model_metrics, b_model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):  # TODO: change the fake_env to add penalty
        # Rollout model using fake_env and add samples into _model_pool

        print('[ Backward Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._backward_rollout_schedule[-1], rollout_batch_size,
        ))

        batch = self.sampler.random_batch(rollout_batch_size)  # sample init states batch from env_pool
        start_obs = batch['observations']

        f_steps_added, b_steps_added = [], []  # to record the num of collected samples

        # perform backward rollout
        obs = start_obs
        for i in range(self._backward_rollout_length):
            act = self._get_before_action(obs)

            before_obs, rew, term, info = self.b_fake_env.step(obs, act, **kwargs)

            samples = {'observations': before_obs, 'actions': act, 'next_observations': obs, 'rewards': rew,
                       'terminals': term}
            self._model_pool.add_samples(samples)  # _model_pool is to add samples get from unrolling the learned dynamic model
            b_steps_added.append(len(obs))

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(),
                                                                              nonterm_mask.shape))
                break
            obs = before_obs[nonterm_mask]

        # perform forward rollout
        print('[ Forward Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {} '.format(
            self._epoch, self._forward_rollout_schedule[-1], rollout_batch_size,
        ))
        obs = start_obs
        for i in range(self._forward_rollout_length):
            act = self._policy.actions_np(obs)

            next_obs, rew, term, info = self.f_fake_env.step(obs, act, **kwargs)

            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew,
                       'terminals': term}
            self._model_pool.add_samples(samples)
            f_steps_added.append(len(obs))

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(),
                                                                              nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        f_mean_rollout_length, b_mean_rollout_length = sum(f_steps_added) / rollout_batch_size, sum(b_steps_added) / rollout_batch_size

        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Total Length: {} | Train rep: {}'.format(
            (self._forward_rollout_length + self._backward_rollout_length) * rollout_batch_size, self._model_pool.size,
            self._model_pool._max_size, f_mean_rollout_length+b_mean_rollout_length, self._n_train_repeat
        ))

        return {'f_mean_rollout_length': f_mean_rollout_length}, {'b_mean_rollout_length': b_mean_rollout_length} # for logging


    def _training_batch(self, batch_size=None):
        # get the training data used for policy and critic update. Called in self._do_training_repeat()
        # batch samples are from both env_pool and model_pool accroding to self._real_ratio

        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size * self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        env_batch = self._pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)

            keys = env_batch.keys()
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                               "loss", "gradients", "gradient_norm", "global_gradient_norm"
                           ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)
        self._actions = actions

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        self._value = tf.reduce_mean(Q_log_targets, axis=0)
        self._target_value = tf.reduce_mean(tuple(
            Q([self._observations_ph, actions])
            for Q in self._Q_targets), axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                    alpha * log_pis
                    - min_Q_log_target
                    - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target()

    def _update_target(self):
        tau = self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops.
        Called by the self._do_training_repeats
        """

        # self._training_progress.update()
        # self._training_progress.set_description()

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict) # update policy and critic

        # update target_Q according to self._tau
        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _do_training_repeats(self, timestep, backward_policy_train_repeat = 1):
        """Repeat training _n_train_repeat times every _train_every_n_steps,
        This method overrides the method in softlearning, since it needs to take care of the backward policy
        """
        print('-------------- into do training repeat -----------', "train_steps this epoch", self._train_steps_this_epoch, 'max * timestep', self._max_train_repeat_per_timestep*self._timestep, self._max_train_repeat_per_timestep, self._timestep)
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        print(' -------------- trained_enough ----------')
        if trained_enough: return

        # Train forward policy and Q
        self._n_train_repeat = 1 # TODO : REMOVE THIS
        for i in range(self._n_train_repeat):
            print('------------ Train forward policy and Q function')
            self._do_training(
                iteration=timestep,
                batch=self._training_batch())

        # train backward policy -- via maximal likelihood. s'->a
        for i in range(backward_policy_train_repeat):  #TODO: Change the backward_policy_train_repeat, Now forward train 1000 times while backward only train 1 time
            """ Our goal is to make the backward rollouts resemble the real trajectory sampled by the current forward policy.Thus
            when training the backward policy, we only use the recent trajectories sampled by the agent in the real environment."""
            batch = self._pool.last_n_random_batch(last_n=self._epoch_length * self._last_n_epoch, batch_size=256)  # TODO: This is incorrect, it uses the recent traj sampled from the real env.
            next_observations = np.array(batch['next_observations'])
            actions = np.array(batch['actions'])
            feed_dict = {
                self._actions_ph: actions,
                self._next_observations_ph: next_observations,
            }
            self._session.run(self._backward_policy_optimizer, feed_dict)

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat


    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
