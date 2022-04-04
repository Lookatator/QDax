from typing import Any, List

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

Array = Any


@flax.struct.dataclass
class Repertoire:
    archive: List
    fitness: Array
    bd: Array
    grid_shape: Array
    min: np.float64
    max: np.float64
    num_indivs: int
    indiv_indices: Array

    @classmethod
    def create(cls, policy_params, max, min, grid_shape):
        grid_shape = jnp.array(grid_shape)
        num_indivs = 0
        indiv_indices = jnp.array([])

        bd = jnp.zeros(grid_shape)
        fitness = jnp.full(grid_shape, jnp.nan)
        # NOTE only 2D atm
        archive = jax.tree_map(
            lambda x: jnp.zeros(
                jnp.repeat(
                    jnp.expand_dims(x, axis=0), jnp.prod(grid_shape), axis=0
                ).shape
            ),
            policy_params,
        )
        return cls(
            archive, fitness, bd, grid_shape, min, max, num_indivs, indiv_indices
        )

    @staticmethod
    def binning(normed, shape):
        return tuple(jnp.multiply(normed, shape - 1).astype(int))

    @staticmethod
    def add_to_archive(repertoire, pop_p, bds, eval_scores, dead):

        normalized_bds = (bds - repertoire.min) / (
            repertoire.max - repertoire.min
        )  # Normlalized BD should be between zero and 1
        bd_cells = jit(jax.vmap(Repertoire.binning, in_axes=(0, None), out_axes=0))(
            normalized_bds, repertoire.grid_shape
        )
        # print(bd_cells)
        bd_indexes = jnp.ravel_multi_index(bd_cells, repertoire.bd.shape, mode="clip")
        maximum_fitness = jax.ops.segment_max(
            eval_scores, bd_indexes, num_segments=repertoire.fitness.ravel().shape[0]
        )
        eval_scores_filtered = jnp.where(
            maximum_fitness.at[bd_indexes].get() == eval_scores,
            eval_scores,
            np.iinfo(np.int32).min,
        )
        # Checking Conditions for fitness function
        current_fitness = repertoire.fitness.ravel().at[bd_indexes].get()
        # Checking if fitness function is nan or not, since nan means we do not have
        # an individual yet
        current_fitness_nan = jnp.isnan(current_fitness)
        # Checking if fitness that we have is better than the one we observed
        better_fitness = current_fitness < eval_scores_filtered

        # NOTE We need to check if two individuals have the same bd and different fitness!!
        # Adding both boolean arrays to perform an OR
        to_be_added = better_fitness + current_fitness_nan

        # We Apply the Mask to remove dead individuals
        to_be_added = jnp.where(dead, False, to_be_added)

        # Every Individual that is not valid will be assigned index 100000 because we
        # cannot cut our arrays. Jit needs to know the size of the array.
        # When adding, every individual will be clipped and sent to the same location
        mult_to_be_added = jnp.where(to_be_added, 1, 100000)

        bd_insertion = bd_indexes * mult_to_be_added

        # Adding individuals indivs to grid
        leaves = []
        for i, weight in enumerate(jax.tree_leaves(pop_p)):
            leaf = jax.tree_leaves(repertoire.archive)[i].at[bd_insertion].set(weight)
            leaves.append(leaf)

        # replacing grid with new leaves that have the updated weights
        new_archive = jax.tree_unflatten(jax.tree_structure(repertoire.archive), leaves)

        new_fitness = jnp.reshape(
            repertoire.fitness.ravel().at[bd_insertion].set(eval_scores),
            repertoire.fitness.shape,
        )
        # print(repertoire.fitness)
        num_indivs = (jnp.where(~jnp.isnan(new_fitness), 1, 0)).sum()

        # returning this to make it jit friendly
        return repertoire.replace(
            archive=new_archive, fitness=new_fitness, num_indivs=num_indivs
        )
