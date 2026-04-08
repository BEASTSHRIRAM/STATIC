"""Task grader for NeoVentEnv episodes."""

from env.neovent_env import NeoVentEnv
from models import EnvironmentState


class TaskGrader:
    """
    Scores a completed episode 0.0-1.0.
    Deterministic given the same episode history.
    """

    @staticmethod
    def grade(env: NeoVentEnv, episode_history: list[dict]) -> float:
        """
        Grade an episode based on performance across multiple metrics.

        Args:
            env: The environment instance (with final state)
            episode_history: List of step records with vitals and settings

        Returns:
            Score from 0.0 to 1.0
        """
        state = env.state()
        steps = len(episode_history)
        if steps == 0:
            return 0.0

        # --- Metric 1: SpO2 time in target (most important) ---
        in_target = sum(1 for h in episode_history
                       if 91 <= h["vitals"].spo2 <= 95)
        spo2_score = in_target / steps  # 0.0-1.0

        # --- Metric 2: No critical hypoxia events ---
        critical_events = sum(1 for h in episode_history
                             if h["vitals"].spo2 < 80)
        hypoxia_penalty = min(critical_events * 0.15, 0.4)

        # --- Metric 3: Lung protection (barotrauma) ---
        final_baro = state.lung_mechanics.compliance_ml_per_cmh2o
        initial_baro = episode_history[0].get("initial_compliance", final_baro)
        compliance_preserved = min(final_baro / max(initial_baro, 0.01), 1.0)
        lung_score = compliance_preserved * 0.3  # Up to 0.3 contribution

        # --- Metric 4: FiO2 weaning (avoid unnecessary oxygen) ---
        avg_fio2 = sum(h["settings"].fio2 for h in episode_history) / steps
        fio2_score = max(0, (0.6 - avg_fio2) / 0.39) * 0.2  # Up to 0.2

        # --- Termination bonus/penalty ---
        if state.termination_reason == "critical_hypoxia":
            termination_mod = -0.3
        elif state.termination_reason == "severe_barotrauma":
            termination_mod = -0.2
        else:
            termination_mod = 0.0

        total = (spo2_score * 0.5) + lung_score + fio2_score \
               - hypoxia_penalty + termination_mod

        return round(float(max(0.0, min(1.0, total))), 3)
