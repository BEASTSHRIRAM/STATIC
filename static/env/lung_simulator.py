"""
Lung physics simulator for neonatal mechanical ventilation.

Single-compartment balloon model of neonatal lung mechanics.
Uses real clinical physics equations from respiratory physiology.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LungState:
    """Internal lung mechanics state."""
    compliance: float        # mL / cmH2O
    resistance: float        # cmH2O / (L/s)
    frc: float               # mL (Functional residual capacity)
    surfactant: float        # 0.0-1.0
    time_step: int = 0       # Current step
    patient_weight_kg: float = 1.0


class LungSimulator:
    """
    Single-compartment balloon model of neonatal lung mechanics.

    Physics equations used:
      Tidal Volume:  Vt = C * (PIP - PEEP)
      Time constant: tau = R * C  (seconds)
      SpO2 from FiO2 and Vt using simplified oxygen dissociation curve
      pCO2 from alveolar ventilation: pCO2 = (VCO2 * k) / Va
        where Va = (Vt - dead_space) * RR

    Compliance drifts over time based on:
      - Surfactant level (low surfactant -> lower compliance)
      - Cumulative volutrauma (over-inflation reduces compliance)
      - Atelecttrauma (repeated collapse increases stiffness)
    """

    def __init__(self, state: LungState):
        self.state = state
        self.cumulative_volutrauma = 0.0
        self.cumulative_atelecttrauma = 0.0

    def step(self, pip: float, peep: float, fio2: float, rr: int, update_state: bool = True) -> dict:
        """
        Given ventilator settings, compute what the baby's vitals will be.

        Args:
            pip: Peak Inspiratory Pressure (cmH2O)
            peep: Positive End-Expiratory Pressure (cmH2O)
            fio2: Fraction of inspired oxygen (0.0-1.0)
            rr: Respiratory rate (breaths/min)
            update_state: Whether to apply lung injury/recovery progression

        Returns:
            dict with spo2, pco2, hr, vt_ml and updated lung mechanics
        """
        # --- Tidal Volume (balloon model) ---
        driving_pressure = pip - peep
        vt = self.state.compliance * driving_pressure
        vt = max(0.5, vt)  # Cannot be negative

        # --- Alveolar Ventilation ---
        anatomical_dead_space_ml = 2.2 * self.state.patient_weight_kg
        va_per_breath = max(0, vt - anatomical_dead_space_ml)
        minute_ventilation = va_per_breath * rr  # mL/min

        # --- pCO2 (simplified Bohr equation) ---
        vco2_ml_per_min = 6.0 * self.state.patient_weight_kg
        pco2 = (vco2_ml_per_min * 0.863) / (minute_ventilation / 1000.0 + 1e-6)
        pco2 = np.clip(pco2, 20, 100)

        # --- SpO2 from FiO2 and PEEP (simplified) ---
        pao2 = (fio2 * 713) - (pco2 / 0.8)         # Alveolar gas equation
        peep_recruitment_bonus = (peep - 4.0) * 1.5  # PEEP recruits alveoli
        effective_pao2 = pao2 + peep_recruitment_bonus
        # Oxygen transfer worsens with low surfactant and poor compliance (V/Q mismatch proxy).
        compliance_factor = np.clip(self.state.compliance / 1.0, 0.35, 1.0)
        transfer_efficiency = (0.25 + 0.75 * self.state.surfactant) * compliance_factor
        effective_pao2 = max(1.0, effective_pao2 * transfer_efficiency)
        # Hill-type O2 dissociation curve (adult-like p50 approximation for stability).
        hill_n = 2.7
        p50 = 35.0
        spo2 = 100 * (effective_pao2 ** hill_n) / (effective_pao2 ** hill_n + p50 ** hill_n)
        spo2 = np.clip(spo2, 40, 100)

        # --- Heart Rate response ---
        base_hr = 140
        if spo2 < 88:
            hr = base_hr + int((88 - spo2) * 3)    # Tachycardia during hypoxia
        elif spo2 > 97:
            hr = base_hr - 10                        # Mild bradycardia at hyperoxia
        else:
            hr = base_hr + (np.random.randint(-8, 8) if update_state else 0)  # Normal variation
        hr = int(np.clip(hr, 60, 220))

        # --- Drift lung mechanics forward ---
        if update_state:
            self._update_compliance(vt, peep, pip)

        return {
            "spo2": round(float(spo2), 1),
            "pco2": round(float(pco2), 1),
            "hr": hr,
            "vt_ml": round(float(vt), 2),
        }

    def _update_compliance(self, vt: float, peep: float, pip: float):
        """
        Compliance drifts each step based on injury accumulation.
        This is the core credit assignment challenge for the RL agent.
        """
        # Volutrauma: each step with Vt > 8 mL/kg damages compliance
        if vt / self.state.patient_weight_kg > 8.0:
            self.cumulative_volutrauma += 0.002
            self.state.compliance *= (1 - 0.001)

        # Atelecttrauma: PEEP too low causes collapse/re-open injury
        if peep < 3.5 and self.state.surfactant < 0.5:
            self.cumulative_atelecttrauma += 0.003
            self.state.compliance *= (1 - 0.0015)

        # Natural recovery: slight compliance improvement over time
        # (represents maturation and surfactant therapy response)
        self.state.compliance += 0.0001
        self.state.compliance = np.clip(self.state.compliance, 0.1, 5.0)
