import functools
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from qdax.core.containers.mapelites_repertoire import GridRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.tasks.arm import arm_scoring_function
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import plot_multidimensional_map_elites_grid

start = time.time()

seed = 42
num_param_dimensions = 8  # num DoF arm
init_batch_size = 100
batch_size = 2048
num_iterations = 1000
grid_shape = (100, 100)
min_param = 0.0
max_param = 1.0
min_bd = 0.0
max_bd = 1.0

# Init a random key
random_key = jax.random.PRNGKey(seed)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
init_variables = jax.random.uniform(
    subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Define emitter
variation_fn = functools.partial(
    isoline_variation,
    iso_sigma=0.005,
    line_sigma=0,
    minval=min_param,
    maxval=max_param,
)
mixing_emitter = MixingEmitter(
    mutation_fn=lambda x, y: (x, y),
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=batch_size,
)

# Define a metrics function
metrics_fn = functools.partial(
    default_qd_metrics,
    qd_offset=0.0,
)

# Instantiate MAP-Elites
empty_repertoire = GridRepertoire.create_empty_repertoire(
    example_batch_genotypes=init_variables,
    min_desc_array=jnp.array([0.0, 0.0]),
    max_desc_array=jnp.array([1.0, 1.0]),
    grid_shape=grid_shape,
)

map_elites = MAPElites(
    scoring_function=arm_scoring_function,
    emitter=mixing_emitter,
    metrics_function=metrics_fn,
)


# Initializes repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init_repertoire(
    init_variables,
    empty_repertoire,
    random_key,
)

# Run MAP-Elites loop
for index_iteration in range(num_iterations):
    (repertoire, emitter_state, metrics, random_key,) = map_elites.update(
        repertoire,
        emitter_state,
        random_key,
    )
    if index_iteration == 0:
        start = time.time()

print(f"Time elapsed after compilation of map_elites.update: {time.time() - start}")

print("Plotting...")
# plot archive
fig, axes = plot_multidimensional_map_elites_grid(
    repertoire=repertoire,
    grid_shape=grid_shape,
    minval=np.array([min_bd, min_bd]),
    maxval=np.array([max_bd, max_bd]),
    vmin=-0.2,
    vmax=0.0,
)
plt.show()
