# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for NeoVentEnv — Neonatal Mechanical Ventilator Management Environment.

Real premature babies die from poor ventilator management. This environment models
that reality with patient-specific lung physics calibrated to the BPD-Neo MRI dataset.
"""

from enum import Enum
from typing import Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class VentMode(str, Enum):
    """Ventilation modes supported by the simulated ventilator."""
    SIMV = "SIMV"   # Synchronized Intermittent Mandatory Ventilation
    HFOV = "HFOV"   # High Frequency Oscillatory Ventilation (hard task only)
    CPAP = "CPAP"   # Continuous Positive Airway Pressure (weaning)


class BPDGrade(str, Enum):
    """Bronchopulmonary Dysplasia severity grades."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class VentilatorSettings(BaseModel):
    """Current ventilator configuration."""
    pip: float  # Peak Inspiratory Pressure cmH2O — range 10-30
    peep: float  # Positive End-Expiratory Pressure cmH2O — range 3-8
    fio2: float  # Fraction of inspired oxygen — range 0.21-1.0
    rr: int    # Respiratory rate breaths/min — range 20-80
    mode: VentMode = VentMode.SIMV


class BabyVitals(BaseModel):
    """Baby's physiological measurements at current step."""
    spo2: float  # Oxygen saturation % — target 91-95
    pco2: float  # Arterial CO2 mmHg — target 45-55
    hr: int    # Heart rate bpm — normal 120-160
    vt_ml: float  # Tidal volume mL — target 4-6 mL/kg


class LungMechanics(BaseModel):
    """Hidden state — agent cannot see these directly in observation."""
    compliance_ml_per_cmh2o: float   # How easily lung stretches
    resistance_cmh2o_per_lps: float  # Airway resistance
    frc_ml: float                    # Functional residual capacity
    surfactant_index: float          # 0.0 (none) to 1.0 (normal)


class PatientProfile(BaseModel):
    """Static patient characteristics."""
    patient_id: str
    gestational_age_weeks: float     # 24-36 weeks
    birth_weight_grams: float        # 400-2500g
    days_of_life: int                # Age at episode start
    bpd_grade: BPDGrade
    lung_volume_ml: float            # Derived from MRI segmentation
    trachea_diameter_mm: float       # Derived from MRI segmentation


class NeoVentAction(Action):
    """Action for NeoVentEnv — ventilator control deltas."""
    delta_pip: float = 0.0   # Must be in {-2,-1,0,1,2}
    delta_peep: float = 0.0  # Must be in {-1,-0.5,0,0.5,1}
    delta_fio2: float = 0.0  # Must be in {-0.05,-0.02,0,0.02,0.05}
    delta_rr: int = 0        # Must be in {-4,-2,0,2,4}
    reasoning: Optional[str] = Field(default=None, description="Agent's chain of thought")


class NeoVentObservation(Observation):
    """Observation from NeoVentEnv — what the agent sees."""
    patient: PatientProfile
    vitals: BabyVitals
    current_settings: VentilatorSettings
    step_number: int = Field(description="Current step in episode")
    time_on_vent_hrs: float = Field(description="Hours on mechanical ventilation")
    cumulative_barotrauma_index: float = Field(description="Lung injury accumulation (0.0-1.0)")
    alarm_flags: List[str] = Field(default_factory=list, description="Active clinical alarms")
    context: Dict = Field(default_factory=dict, description="Task-specific context")


class EnvironmentState(BaseModel):
    """Full internal state exposed by state() method."""
    task_id: str
    step_number: int
    patient: PatientProfile
    lung_mechanics: LungMechanics    # Full hidden state exposed by state()
    current_settings: VentilatorSettings
    vitals: BabyVitals
    cumulative_reward: float
    done: bool
    termination_reason: Optional[str] = None   # Why episode ended
