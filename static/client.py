# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NeoVentEnv Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import NeoVentAction, NeoVentObservation


class NeoVentEnvClient(
    EnvClient[NeoVentAction, NeoVentObservation, State]
):
    """
    Client for the NeoVentEnv Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with NeoVentEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="task_easy")
        ...     print(result.observation.vitals.spo2)
        ...
        ...     action = NeoVentAction(delta_fio2=-0.02)
        ...     result = client.step(action)
        ...     print(result.observation.vitals.spo2)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = NeoVentEnvClient.from_docker_image("neovent:latest")
        >>> try:
        ...     result = client.reset(task_id="task_easy")
        ...     action = NeoVentAction(delta_fio2=-0.02)
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: NeoVentAction) -> Dict:
        """
        Convert NeoVentAction to JSON payload for step message.

        Args:
            action: NeoVentAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "delta_pip": action.delta_pip,
            "delta_peep": action.delta_peep,
            "delta_fio2": action.delta_fio2,
            "delta_rr": action.delta_rr,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[NeoVentObservation]:
        """
        Parse server response into StepResult[NeoVentObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with NeoVentObservation
        """
        obs_data = payload.get("observation", {})
        observation = NeoVentObservation(
            patient=obs_data.get("patient", {}),
            vitals=obs_data.get("vitals", {}),
            current_settings=obs_data.get("current_settings", {}),
            step_number=obs_data.get("step_number", 0),
            time_on_vent_hrs=obs_data.get("time_on_vent_hrs", 0.0),
            cumulative_barotrauma_index=obs_data.get("cumulative_barotrauma_index", 0.0),
            alarm_flags=obs_data.get("alarm_flags", []),
            context=obs_data.get("context", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
