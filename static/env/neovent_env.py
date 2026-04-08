"""
NeoVentEnv — OpenEnv compliant neonatal ventilator environment.

Main environment interface for the NeoVentEnv simulation.
"""

import random
import numpy as np
from typing import Optional, Tuple
from models import (
    NeoVentAction, NeoVentObservation, VentilatorSettings, BabyVitals,
    PatientProfile, BPDGrade, LungMechanics, EnvironmentState
)
from env.lung_simulator import LungSimulator, LungState
from env.patient_loader import load_patients


class NeoVentEnv:
    """
    NeoVentEnv — OpenEnv compliant neonatal ventilator environment.

    Interface:
        obs          = env.reset(task_id)
        obs, rew, done, info = env.step(action)
        state        = env.state()
    """

    VALID_TASKS = ["task_easy", "task_medium", "task_hard"]
    MAX_STEPS = {"task_easy": 40, "task_medium": 80, "task_hard": 120}

    def __init__(self, data_path: str = "data/patients.csv"):
        self.patients = load_patients(data_path)
        self._reset_internal()

    def _reset_internal(self):
        """Reset all internal state variables."""
        self.task_id = None
        self.step_num = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.patient_profile = None
        self.simulator = None
        self.current_settings = None
        self.termination_reason = None
        self._last_vitals = None
        self._last_baro_idx = 0.0

    def reset(self, task_id: str = "task_easy") -> NeoVentObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of "task_easy", "task_medium", "task_hard"

        Returns:
            Initial observation
        """
        assert task_id in self.VALID_TASKS, f"Unknown task: {task_id}"
        self._reset_internal()
        self.task_id = task_id

        # Select patient appropriate for task difficulty
        patient_data = self._select_patient(task_id)
        self.patient_profile = patient_data["profile"]
        self.simulator = LungSimulator(patient_data["lung_state"])

        # Starting ventilator settings
        self.current_settings = self._initial_settings(task_id)

        return self._build_observation()

    def step(self, action: NeoVentAction) -> Tuple[NeoVentObservation, dict, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: NeoVentAction with delta adjustments

        Returns:
            (observation, reward_dict, done, info)
        """
        assert not self.done, "Episode is over. Call reset() first."
        assert self.task_id is not None, "Call reset() before step()."

        # Validate and clip action deltas to safe bounds
        action = self._validate_action(action)

        # Apply deltas to current settings
        self.current_settings = VentilatorSettings(
            pip=float(np.clip(self.current_settings.pip + action.delta_pip, 10, 35)),
            peep=float(np.clip(self.current_settings.peep + action.delta_peep, 2, 10)),
            fio2=float(np.clip(self.current_settings.fio2 + action.delta_fio2, 0.21, 1.0)),
            rr=int(np.clip(self.current_settings.rr + action.delta_rr, 15, 80)),
            mode=self.current_settings.mode,
        )

        # Simulate baby's response
        vitals_dict = self.simulator.step(
            pip=self.current_settings.pip,
            peep=self.current_settings.peep,
            fio2=self.current_settings.fio2,
            rr=self.current_settings.rr,
        )
        vitals = BabyVitals(**vitals_dict)

        # Compute reward
        reward_dict = self._compute_reward(vitals, self.current_settings)

        # Check terminal conditions
        self.step_num += 1
        self.cumulative_reward += reward_dict["total"]
        self._check_termination(vitals)

        obs = self._build_observation(vitals)
        return obs, reward_dict, self.done, {"step": self.step_num}

    def state(self) -> EnvironmentState:
        """Returns full internal state including hidden lung mechanics."""
        ls = self.simulator.state
        return EnvironmentState(
            task_id=self.task_id,
            step_number=self.step_num,
            patient=self.patient_profile,
            lung_mechanics=LungMechanics(
                compliance_ml_per_cmh2o=ls.compliance,
                resistance_cmh2o_per_lps=ls.resistance,
                frc_ml=ls.frc,
                surfactant_index=ls.surfactant,
            ),
            current_settings=self.current_settings,
            vitals=self._last_vitals,
            cumulative_reward=self.cumulative_reward,
            done=self.done,
            termination_reason=self.termination_reason,
        )

    def _compute_reward(self, vitals: BabyVitals, settings: VentilatorSettings) -> dict:
        """
        Reward is continuous every step — not binary at episode end.

        Components:
          spo2_score:      +1.0 if in target, graded if close, -1.0 if hypoxic
          gentleness:      +0.5 if Vt within safe range per kg
          barotrauma_cost: -0.3 * cumulative_barotrauma_index per step
          hyperoxia_cost:  -0.2 if FiO2 > 0.6 while SpO2 > 95 (unnecessary O2)
          hypoxia_penalty: -2.0 if SpO2 < 85 (sharp cliff, life-threatening)
        """
        weight_kg = self.patient_profile.birth_weight_grams / 1000.0
        breakdown = {}

        # SpO2 target: 91-95%
        if 91 <= vitals.spo2 <= 95:
            breakdown["spo2_score"] = 1.0
        elif 88 <= vitals.spo2 < 91 or 95 < vitals.spo2 <= 97:
            breakdown["spo2_score"] = 0.4   # Partial credit
        elif vitals.spo2 < 88:
            breakdown["spo2_score"] = -1.0  # Hypoxia
        else:
            breakdown["spo2_score"] = -0.3  # Hyperoxia (above 97)

        # Hard cliff for severe hypoxia
        if vitals.spo2 < 85:
            breakdown["hypoxia_penalty"] = -2.0
        else:
            breakdown["hypoxia_penalty"] = 0.0

        # Gentleness: tidal volume per kg
        vt_per_kg = vitals.vt_ml / weight_kg
        if 4.0 <= vt_per_kg <= 6.0:
            breakdown["gentleness"] = 0.5
        elif vt_per_kg > 8.0:
            breakdown["gentleness"] = -0.5  # Volutrauma risk
        else:
            breakdown["gentleness"] = 0.1

        # Unnecessary oxygen is harmful (retinopathy of prematurity)
        if settings.fio2 > 0.6 and vitals.spo2 > 95:
            breakdown["hyperoxia_cost"] = -0.2
        else:
            breakdown["hyperoxia_cost"] = 0.0

        # Barotrauma accumulation cost
        baro_idx = self.simulator.cumulative_volutrauma + self.simulator.cumulative_atelecttrauma
        breakdown["barotrauma_cost"] = -0.3 * min(baro_idx, 1.0)

        total = sum(breakdown.values())
        self._last_baro_idx = baro_idx
        
        return {
            "total": total,
            "reward": total,  # OpenEnv standard field
            "breakdown": breakdown,
            "cumulative": self.cumulative_reward + total,
        }

    def _check_termination(self, vitals: BabyVitals):
        """Check if episode should terminate."""
        max_steps = self.MAX_STEPS[self.task_id]
        if self.step_num >= max_steps:
            self.done = True
            self.termination_reason = "max_steps_reached"
        elif vitals.spo2 < 75:
            self.done = True
            self.termination_reason = "critical_hypoxia"
        elif self.simulator.cumulative_volutrauma > 0.8:
            self.done = True
            self.termination_reason = "severe_barotrauma"

    def _select_patient(self, task_id: str) -> dict:
        """Select a patient appropriate for the task difficulty."""
        bpd_filter = {
            "task_easy": [BPDGrade.NONE, BPDGrade.MILD],
            "task_medium": [BPDGrade.MILD, BPDGrade.MODERATE],
            "task_hard": [BPDGrade.MODERATE, BPDGrade.SEVERE],
        }
        eligible = [p for p in self.patients
                   if p["profile"].bpd_grade in bpd_filter[task_id]]
        if not eligible:
            eligible = self.patients
        return random.choice(eligible)

    def _initial_settings(self, task_id: str) -> VentilatorSettings:
        """Get initial ventilator settings for the task."""
        defaults = {
            "task_easy": dict(pip=20, peep=6, fio2=0.30, rr=45),
            "task_medium": dict(pip=23, peep=7, fio2=0.45, rr=55),
            "task_hard": dict(pip=26, peep=7, fio2=0.60, rr=62),
        }
        settings_dict = defaults[task_id]
        return VentilatorSettings(**settings_dict)

    def _build_observation(self, vitals: Optional[BabyVitals] = None) -> NeoVentObservation:
        """Build observation from current state."""
        if vitals is None:
            # Initial observation — compute vitals from starting settings
            vd = self.simulator.step(
                self.current_settings.pip,
                self.current_settings.peep,
                self.current_settings.fio2,
                self.current_settings.rr,
                update_state=False,
            )
            vitals = BabyVitals(**vd)

        self._last_vitals = vitals
        alarms = []
        if vitals.spo2 < 88:
            alarms.append("SPO2_LOW")
        if vitals.spo2 > 97:
            alarms.append("SPO2_HIGH")
        if vitals.pco2 > 60:
            alarms.append("HYPERCAPNIA")
        if vitals.pco2 < 35:
            alarms.append("HYPOCAPNIA")
        if self.current_settings.pip > 28:
            alarms.append("HIGH_PEAK_PRESSURE")

        return NeoVentObservation(
            patient=self.patient_profile,
            vitals=vitals,
            current_settings=self.current_settings,
            step_number=self.step_num,
            time_on_vent_hrs=self.step_num * 0.25,  # Each step = 15 min
            cumulative_barotrauma_index=min(
                self.simulator.cumulative_volutrauma + self.simulator.cumulative_atelecttrauma, 1.0),
            alarm_flags=alarms,
            context={"task_id": self.task_id},
            reward=0.0,  # Will be set by step()
            done=self.done,
        )

    def _validate_action(self, action: NeoVentAction) -> NeoVentAction:
        """Validate and snap action deltas to valid ranges."""
        valid_pip = {-2, -1, 0, 1, 2}
        valid_peep = {-1, -0.5, 0, 0.5, 1}
        valid_fio2 = {-0.05, -0.02, 0, 0.02, 0.05}
        valid_rr = {-4, -2, 0, 2, 4}

        def snap(val, valid_set):
            return min(valid_set, key=lambda x: abs(x - val))

        return NeoVentAction(
            delta_pip=snap(action.delta_pip, valid_pip),
            delta_peep=snap(action.delta_peep, valid_peep),
            delta_fio2=snap(action.delta_fio2, valid_fio2),
            delta_rr=snap(action.delta_rr, valid_rr),
            reasoning=action.reasoning,
        )
