import functools
from typing import Callable, Optional

import brax

from qdax.algorithms.brax_envs.exploration_wrappers import MazeWrapper, TrapWrapper
from qdax.algorithms.brax_envs.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)
from qdax.algorithms.brax_envs.pointmaze import PointMaze

_parl_envs = {
    "pointmaze": PointMaze,
}

_parl_custom_envs = {
    "anttrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, TrapWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}, {}],
    },
    "antmaze": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, MazeWrapper],
        "kwargs": [{"minval": [-5.0, -5.0], "maxval": [40.0, 40.0]}, {}],
    },
    "ant_uni": {"env": "ant", "wrappers": [FeetContactWrapper], "kwargs": [{}, {}]},
    "humanoid_uni": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "hopper_uni": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "humanoid_omni": {
        "env": "humanoid",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> brax.envs.Env:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in brax.envs._envs.keys():
        env = brax.envs._envs[env_name](**kwargs)
    elif env_name in _parl_envs.keys():
        env = _parl_envs[env_name](**kwargs)
    elif env_name in _parl_custom_envs.keys():
        base_env_name = _parl_custom_envs[env_name]["env"]
        env = brax.envs._envs[base_env_name](**kwargs)

        # roll with parl wrappers
        wrappers = _parl_custom_envs[env_name]["wrappers"]
        kwargs_list = _parl_custom_envs[env_name]["kwargs"]
        for wrapper, kwargs in zip(wrappers, kwargs_list):
            env = wrapper(env, base_env_name, **kwargs)
    else:
        raise NotImplementedError("This environment name does not exist!")

    if episode_length is not None:
        env = brax.envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = brax.envs.wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = brax.envs.wrappers.AutoResetWrapper(env)

    return env


def create_fn(env_name: str, **kwargs) -> Callable[..., brax.envs.Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)