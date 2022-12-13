# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/examples/pgame.ipynb)

# # Optimizing with PGAME in Jax
#
# This notebook shows how to use QDax to find diverse and performing controllers in MDPs with [Policy Gradient Assisted MAP-Elites](https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf).
# It can be run locally or on Google Colab. We recommand to use a GPU. This notebook will show:
#
# - how to define the problem
# - how to create the PGAME emitter
# - how to create a Map-elites instance
# - which functions must be defined before training
# - how to launch a certain number of training steps
# - how to visualize the results of the training process

# +
#@title Installs and Imports
# !pip install ipympl |tail -n 1
# # %matplotlib widget
# from google.colab import output
# output.enable_custom_widget_manager()

import os

import functools
import time

import jax
import jax.numpy as jnp

import brax

import qdax


from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, \
    compute_euclidean_centroids
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics


# +
#@title QD Training Definitions Fields
#@markdown ---
env_name = 'walker2d_uni'#@param['ant_uni', 'hopper_uni', 'walker_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 1000 #@param {type:"integer"}
num_iterations = 800 #@param {type:"integer"}
seed = 42 #@param {type:"integer"}
policy_hidden_layer_sizes = (64, 64) #@param {type:"raw"}
iso_sigma = 0.01 #@param {type:"number"}
line_sigma = 0.1 #@param {type:"number"}
num_init_cvt_samples = 50000 #@param {type:"integer"}
grid_shape = (50, 50) #@param {type:"integer"}
min_bd = 0. #@param {type:"number"}
max_bd = 1.0 #@param {type:"number"}

#@title PGA-ME Emitter Definitions Fields
proportion_mutation_ga = 0.5

# TD3 params
env_batch_size = 512 #@param {type:"number"}
replay_buffer_size = 1000000 #@param {type:"number"}
critic_hidden_layer_size = (256, 256) #@param {type:"raw"}
critic_learning_rate = 3e-4 #@param {type:"number"}
greedy_learning_rate = 3e-4 #@param {type:"number"}
policy_learning_rate = 1e-3 #@param {type:"number"}
noise_clip = 0.5 #@param {type:"number"}
policy_noise = 0.2 #@param {type:"number"}
discount = 0.99 #@param {type:"number"}
reward_scaling = 1.0 #@param {type:"number"}
transitions_batch_size = 256 #@param {type:"number"}
soft_tau_update = 0.005 #@param {type:"number"}
num_critic_training_steps = 300 #@param {type:"number"}
num_pg_training_steps = 100 #@param {type:"number"}
policy_delay = 2 #@param {type:"number"}
#@markdown ---
# -

# ## Init environment, policy, population params, init states of the env
#
# Define the environment in which the policies will be trained. In this notebook, we focus on controllers learning to move a robot in a physical simulation. We also define the shared policy, that every individual in the population will use. Once the policy is defined, all individuals are defined by their parameters, that corresponds to their genotype.

# +
# Init environment
env = environments.create(env_name, episode_length=episode_length)

# Init a random key
random_key = jax.random.PRNGKey(seed)

# Init policy network
policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
keys = jax.random.split(subkey, num=env_batch_size)
fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

# Create the initial environment states
random_key, subkey = jax.random.split(random_key)
keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=env_batch_size, axis=0)
reset_fn = jax.jit(jax.vmap(env.reset))
init_states = reset_fn(keys)


# -

# ## Define the way the policy interacts with the env

# Define the fonction to play a step with the policy in the environment
def play_step_fn(
  env_state,
  policy_params,
  random_key,
):
    """
    Play an environment step and return the updated state and the transition.
    """

    actions = policy_network.apply(policy_params, env_state.obs)
    
    state_desc = env_state.info["state_descriptor"]
    next_state = env.step(env_state, actions)

    transition = QDTransition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        truncations=next_state.info["truncation"],
        state_desc=state_desc,
        next_state_desc=next_state.info["state_descriptor"],
    )

    return next_state, policy_params, random_key, transition


# ## Define the scoring function and the way metrics are computed
#
# The scoring function is used in the evaluation step to determine the fitness and behavior descriptor of each individual. 

# +
# Prepare the scoring function
bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
scoring_fn = functools.partial(
    scoring_function,
    init_states=init_states,
    episode_length=episode_length,
    play_step_fn=play_step_fn,
    behavior_descriptor_extractor=bd_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)
# -

# ## Define the emitter: PG Emitter
#
# The emitter is used to evolve the population at each mutation step. In this example, the emitter is the Policy Gradient emitter, the one used in Policy Gradient Assisted Map Elites. It trains a critic with the transitions experienced in the environment and uses the critic to apply Policy Gradient updates to the policies evolved.

# Define the PG-emitter config
pga_emitter_config = PGAMEConfig(
    env_batch_size=env_batch_size,
    batch_size=transitions_batch_size,
    proportion_mutation_ga=proportion_mutation_ga,
    critic_hidden_layer_size=critic_hidden_layer_size,
    critic_learning_rate=critic_learning_rate,
    greedy_learning_rate=greedy_learning_rate,
    policy_learning_rate=policy_learning_rate,
    noise_clip=noise_clip,
    policy_noise=policy_noise,
    discount=discount,
    reward_scaling=reward_scaling,
    replay_buffer_size=replay_buffer_size,
    soft_tau_update=soft_tau_update,
    num_critic_training_steps=num_critic_training_steps,
    num_pg_training_steps=num_pg_training_steps,
    policy_delay=policy_delay,
)

# +
# Get the emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)

pg_emitter = PGAMEEmitter(
    config=pga_emitter_config,
    policy_network=policy_network,
    env=env,
    variation_fn=variation_fn,
)
# -

# ## Instantiate and initialise the MAP Elites algorithm

# +
# Instantiate MAP Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=pg_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids
centroids = compute_euclidean_centroids(
    grid_shape=grid_shape,
    minval=min_bd,
    maxval=max_bd,
)

# compute initial repertoire
repertoire, emitter_state, random_key = map_elites.init(
    init_variables, centroids, random_key
)

# +
log_period = 10
num_loops = int(num_iterations / log_period)

csv_logger = CSVLogger(
    "pgame-logs.csv",
    header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
)
all_metrics = {}

# main loop
map_elites_scan_update = map_elites.scan_update
for i in range(num_loops):
    start_time = time.time()
    # main iterations
    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites_scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=log_period,
    )
    timelapse = time.time() - start_time

    # log metrics
    logged_metrics = {"time": timelapse, "loop": 1+i, "iteration": 1 + i*log_period}
    for key, value in metrics.items():
        # take last value
        logged_metrics[key] = value[-1]

        # take all values
        if key in all_metrics.keys():
            all_metrics[key] = jnp.concatenate([all_metrics[key], value])
        else:
            all_metrics[key] = value

    csv_logger.log(logged_metrics)
    print(i, logged_metrics)

# +
#@title Visualization

# create the x-axis array
env_steps = jnp.arange(num_iterations) * episode_length * env_batch_size

# create the plots and the grid
fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=min_bd, max_bd=max_bd)
# -






