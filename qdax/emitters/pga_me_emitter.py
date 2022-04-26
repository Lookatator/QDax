""" Implements the PGA-ME algorithm in jax for brax environments, based on:
https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import tree_map

from qdax.algorithms.map_elites import MapElitesRepertoire
from qdax.brax_envs.utils_wrappers import QDEnv
from qdax.buffers.buffers import FlatBuffer, QDTransition
from qdax.emitters.emitter import Emitter
from qdax.losses.td3_loss import make_td3_loss_fn
from qdax.networks.flax_networks import QModule
from qdax.types import EmitterState, Genotype, Params, RNGKey


@dataclass
class PGAMEConfig:
    """Configuration for PGAME Algorithm"""

    env_batch_size: int = 100
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005


class PGEmitterState(EmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    greedy_policy_params: Params
    greedy_policy_opt_state: optax.OptState
    controllers_optimizer_state: optax.OptState
    target_critic_params: Params
    target_greedy_policy_params: Params
    replay_buffer: FlatBuffer
    random_key: RNGKey
    steps: jnp.ndarray


class PGEmitter(Emitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm.
    """

    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        crossover_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        self._config = config
        self._env = env
        self._crossover_fn = crossover_fn
        self._policy_network = policy_network

        # Init Critics
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network

        # Set up the losses and optimizers - return the opt states
        policy_loss_fn, critic_loss_fn = make_td3_loss_fn(
            policy_fn=policy_network.apply,
            critic_fn=critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        self._policy_loss_fn = policy_loss_fn
        self._critic_loss_fn = critic_loss_fn

        # Init optimizers
        self._greedy_policy_optimizer = optax.adam(
            learning_rate=self._config.greedy_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._controllers_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )

    def init_fn(self, init_genotypes: Genotype, random_key: RNGKey) -> PGEmitterState:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the PGEmitter.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action
        )
        target_critic_params = tree_map(lambda x: jnp.asarray(x.copy()), critic_params)

        greedy_policy_params = tree_map(
            lambda x: jnp.asarray(x[0].copy()), init_genotypes
        )
        target_greedy_policy_params = tree_map(
            lambda x: jnp.asarray(x[0].copy()), init_genotypes
        )

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        greedy_optimizer_state = self._greedy_policy_optimizer.init(
            greedy_policy_params
        )
        controllers_optimizer_state = self._controllers_optimizer.init(
            greedy_policy_params
        )

        # Initialize replay buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = FlatBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        # Initial training state
        training_state = PGEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            greedy_policy_params=greedy_policy_params,
            greedy_policy_opt_state=greedy_optimizer_state,
            controllers_optimizer_state=controllers_optimizer_state,
            target_critic_params=target_critic_params,
            target_greedy_policy_params=target_greedy_policy_params,
            random_key=random_key,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
        )

        return training_state

    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def emit_fn(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: PGEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, PGEmitterState, RNGKey]:
        """Do a single PGA-ME iteration: train critics and greedy policy,
        make mutations (evo and pg), score solution, fill replay buffer and insert back
        in the MAP-Elites grid.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """

        batch_size = self._config.env_batch_size

        def scan_train_critics(
            carry: PGEmitterState, unused: Any
        ) -> Tuple[PGEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state)
            return new_emitter_state, ()

        # Train critics
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (),
            length=self._config.num_critic_training_steps,
        )

        # Mutation evo
        mutation_ga_batch_size = int(self._config.proportion_mutation_ga * batch_size)
        x1, random_key = repertoire.sample(random_key, mutation_ga_batch_size)
        x2, random_key = repertoire.sample(random_key, mutation_ga_batch_size)
        x_mutation_ga, random_key = self._crossover_fn(x1, x2, random_key)

        # Mutation PG
        mutation_pg_batch_size = int(batch_size - mutation_ga_batch_size - 1)
        x1, random_key = repertoire.sample(random_key, mutation_pg_batch_size)
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        x_mutation_pg = jax.vmap(mutation_fn)(x1)

        # Add dimension for concatenation
        greedy_policy_params = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), emitter_state.greedy_policy_params
        )

        # gather offspring
        genotypes = jax.tree_multimap(
            lambda x, y, z: jnp.concatenate([x, y, z], axis=0),
            x_mutation_ga,
            x_mutation_pg,
            greedy_policy_params,
        )

        return genotypes, emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def state_update_fn(
        self, emitter_state: PGEmitterState, **kwargs: Any
    ) -> PGEmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.

        Here it is used to fill the Replay Buffer with the transitions
        from the scoring of the genotypes.

        Args:
            emitter_state: current emitter state.

        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # get the transitions out of the dictionary
        assert "transitions" in kwargs.keys(), "Missing transitions or wrong key"
        transitions = kwargs["transitions"]

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)

        return emitter_state.replace(replay_buffer=replay_buffer)  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(self, emitter_state: PGEmitterState) -> PGEmitterState:
        """Apply one gradient step to critics and to the greedy policy
        (contained in carry in training_state), then soft update target critics
        and target greedy policy.

        Those updates are very similar to those made in TD3.

        Args:
            emitter_state: actual emitter state

        Returns:
            New emitter state where the critic and the greedy policy have been
            updated. Optimizer states have also been updated in the process.
        """

        # Sample a batch of transitions in the buffer
        key = emitter_state.random_key
        key, subkey = jax.random.split(key)
        replay_buffer = emitter_state.replay_buffer
        samples = replay_buffer.sample(subkey, sample_size=self._config.batch_size)

        # Update Critic
        key, subkey = jax.random.split(key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            emitter_state.critic_params,
            emitter_state.target_greedy_policy_params,
            emitter_state.target_critic_params,
            samples,
            subkey,
        )
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, emitter_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(emitter_state.critic_params, critic_updates)
        # Soft update of target critic network
        target_critic_params = jax.tree_multimap(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            emitter_state.target_critic_params,
            critic_params,
        )

        # Update greedy policy
        key, subkey = jax.random.split(key)
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            emitter_state.greedy_policy_params,
            emitter_state.critic_params,
            samples,
        )
        (
            policy_updates,
            policy_optimizer_state,
        ) = self._greedy_policy_optimizer.update(
            policy_gradient, emitter_state.greedy_policy_opt_state
        )
        greedy_policy_params = optax.apply_updates(
            emitter_state.greedy_policy_params, policy_updates
        )
        # Soft update of target greedy policy
        target_greedy_policy_params = jax.tree_multimap(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            emitter_state.target_greedy_policy_params,
            greedy_policy_params,
        )

        # Create new training state
        new_state = PGEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            greedy_policy_params=greedy_policy_params,
            greedy_policy_opt_state=policy_optimizer_state,
            controllers_optimizer_state=emitter_state.controllers_optimizer_state,
            target_critic_params=target_critic_params,
            target_greedy_policy_params=target_greedy_policy_params,
            random_key=key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_state

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        controller_params: Genotype,
        emitter_state: PGEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.

        Args:
            controller_params: a controller, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.

        Returns:
            the updated params of the neural network.
        """

        def scan_train_controller(
            carry: Tuple[PGEmitterState, Genotype], unused: Any
        ) -> Tuple[Tuple[PGEmitterState, Genotype], Any]:
            emitter_state, controller_params = carry
            (
                new_emitter_state,
                new_controller_params,
            ) = self._train_controller(emitter_state, controller_params)
            return (new_emitter_state, new_controller_params), ()

        (emitter_state, controller_params), _ = jax.lax.scan(
            scan_train_controller,
            (emitter_state, controller_params),
            (),
            length=self._config.num_pg_training_steps,
        )

        return controller_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_controller(
        self,
        emitter_state: PGEmitterState,
        controller_params: Params,
    ) -> Tuple[PGEmitterState, Params]:
        """Apply one gradient step to a policy (called controllers_params).

        Args:
            emitter_state: current state of the emitter.
            controller_params: parameters corresponding to the weights and bias of
                the neural network that defines the controller.

        Returns:
            The new emitter state and new params of the NN.
        """

        # Sample a batch of transitions in the buffer
        key = emitter_state.random_key
        key, subkey = jax.random.split(key)
        replay_buffer = emitter_state.replay_buffer
        samples = replay_buffer.sample(subkey, sample_size=self._config.batch_size)
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            controller_params,
            emitter_state.critic_params,
            samples,
        )
        # Compute gradient and update policies
        (policy_updates, policy_optimizer_state,) = self._controllers_optimizer.update(
            policy_gradient, emitter_state.controllers_optimizer_state
        )
        controller_params = optax.apply_updates(controller_params, policy_updates)

        # Create new training state
        new_emitter_state = PGEmitterState(
            critic_params=emitter_state.critic_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            greedy_policy_params=emitter_state.greedy_policy_params,
            greedy_policy_opt_state=emitter_state.greedy_policy_opt_state,
            controllers_optimizer_state=policy_optimizer_state,
            target_critic_params=emitter_state.target_critic_params,
            target_greedy_policy_params=emitter_state.target_greedy_policy_params,
            random_key=key,
            steps=emitter_state.steps,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state, controller_params