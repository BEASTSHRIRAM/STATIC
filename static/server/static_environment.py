# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NeoVentEnv Environment Implementation.

Neonatal mechanical ventilator management environment.
Agent acts as respiratory therapist for premature babies in the NICU.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NeoVentAction, NeoVentObservation
    from ..env.neovent_env import NeoVentEnv
except ImportError:
    from models import NeoVentAction, NeoVentObservation
    from env.neovent_env import NeoVentEnv


class NeoVentEnvironment(Environment):
    """
    Neonatal ventilator management environment.

    Agent plays the role of a respiratory therapist. At every timestep it
    observes the baby's current physiological state and decides how to adjust
    the ventilator settings. The goal is to keep SpO2 in the safe range
    (91-95%) while minimising lung damage (barotrauma) and oxygen toxicity.

    Real premature babies die from poor ventilator management.
    This environment reflects that weight.

    Example:
        >>> env = NeoVentEnvironment()
        >>> obs = env.reset(task_id="task_easy")
        >>> obs, _, _, _ = env.step(NeoVentAction(delta_fio2=-0.02))
    """

    # Enable concurrent WebSocket sessions.
    # Each client gets their own NeoVentEnv instance.
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    _shared_env = None
    _shared_state = None
    _shared_task_id = "task_easy"
    _shared_reset_count = 0

    def __init__(self, data_path: str = "data/patients.csv"):
        """Initialize the neonatal ventilator environment."""
        if NeoVentEnvironment._shared_env is None:
            NeoVentEnvironment._shared_env = NeoVentEnv(data_path)
        if NeoVentEnvironment._shared_state is None:
            NeoVentEnvironment._shared_state = State(episode_id=str(uuid4()), step_count=0)

        self.neovent_env = NeoVentEnvironment._shared_env
        self._state = NeoVentEnvironment._shared_state
        self._reset_count = NeoVentEnvironment._shared_reset_count
        self.task_id = NeoVentEnvironment._shared_task_id

    def reset(self, task_id: str = "task_easy") -> NeoVentObservation:
        """
        Reset the environment.

        Args:
            task_id: One of "task_easy", "task_medium", "task_hard"

        Returns:
            NeoVentObservation for the initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.task_id = task_id
        NeoVentEnvironment._shared_state = self._state
        NeoVentEnvironment._shared_reset_count = self._reset_count
        NeoVentEnvironment._shared_task_id = self.task_id

        obs = self.neovent_env.reset(task_id)
        return obs

    def step(self, action: NeoVentAction) -> NeoVentObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: NeoVentAction with ventilator adjustments

        Returns:
            NeoVentObservation with updated state
        """
        self._state.step_count += 1
        NeoVentEnvironment._shared_state = self._state

        obs, reward_dict, done, info = self.neovent_env.step(action)
        
        # Merge reward back into observation for OpenEnv compatibility
        obs.reward = reward_dict["total"]
        obs.done = done
        obs.metadata = info

        return obs

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
