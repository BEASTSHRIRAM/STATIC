"""
Smoke tests for NeoVentEnv.
Run with: pytest tests/test_env.py -v
"""

import pytest
from models import NeoVentAction, BPDGrade
from env.neovent_env import NeoVentEnv


class TestNeoVentEnv:
    """Test suite for NeoVentEnv."""

    @pytest.fixture
    def env(self):
        """Create environment for tests."""
        return NeoVentEnv()

    def test_reset_easy(self, env):
        """Test reset on easy task."""
        obs = env.reset("task_easy")
        assert obs is not None
        assert obs.patient is not None
        assert obs.vitals is not None
        assert obs.current_settings is not None
        assert obs.step_number == 0
        assert not obs.done

    def test_reset_medium(self, env):
        """Test reset on medium task."""
        obs = env.reset("task_medium")
        assert obs is not None
        assert obs.patient.bpd_grade in [BPDGrade.MILD, BPDGrade.MODERATE]

    def test_reset_hard(self, env):
        """Test reset on hard task."""
        obs = env.reset("task_hard")
        assert obs is not None
        assert obs.patient.bpd_grade in [BPDGrade.MODERATE, BPDGrade.SEVERE]

    def test_step_basic(self, env):
        """Test basic step functionality."""
        env.reset("task_easy")
        action = NeoVentAction(delta_fio2=-0.02)
        obs, reward_dict, done, info = env.step(action)

        assert obs is not None
        assert obs.step_number == 1
        assert "total" in reward_dict
        assert "reward" in reward_dict
        assert "breakdown" in reward_dict
        assert isinstance(done, bool)
        assert "step" in info

    def test_episode_completion(self, env):
        """Test full episode runs without error."""
        obs = env.reset("task_easy")
        done = False
        step = 0

        while not done and step < 50:
            action = NeoVentAction()  # No-op
            obs, reward_dict, done, info = env.step(action)
            step += 1

        assert step > 0
        assert obs.done == done

    def test_state_exposure(self, env):
        """Test that state() exposes full internal state."""
        env.reset("task_easy")
        obs, _, _, _ = env.step(NeoVentAction())

        state = env.state()
        assert state.task_id == "task_easy"
        assert state.step_number == 1
        assert state.patient is not None
        assert state.lung_mechanics is not None
        assert state.vitals is not None
        assert isinstance(state.cumulative_reward, float)

    def test_action_clipping(self, env):
        """Test that actions are clipped to valid ranges."""
        env.reset("task_easy")
        # Request extremely large changes
        action = NeoVentAction(delta_pip=100, delta_fio2=1.0)
        obs, _, _, _ = env.step(action)

        # Settings should be clipped
        assert obs.current_settings.pip <= 35
        assert obs.current_settings.fio2 <= 1.0

    def test_hypoxia_penalty(self, env):
        """Test that severe hypoxia triggers negative reward."""
        env.reset("task_easy")
        # Attempt to cause hypoxia by removing all oxygen
        for _ in range(10):
            action = NeoVentAction(delta_fio2=-0.05)
            obs, reward_dict, done, info = env.step(action)
            if obs.vitals.spo2 < 85:
                assert reward_dict["breakdown"]["hypoxia_penalty"] < 0
                break

    def test_invalid_task_id(self, env):
        """Test that invalid task ID raises error."""
        with pytest.raises(AssertionError):
            env.reset("invalid_task")

    def test_step_before_reset(self, env):
        """Test that step() before reset() raises error."""
        with pytest.raises(AssertionError):
            env.step(NeoVentAction())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
