"""Load BPD-Neo patient data and initialize lung mechanics."""

import pandas as pd
import numpy as np
from models import PatientProfile, BPDGrade
from env.lung_simulator import LungState


BPD_GRADE_COMPLIANCE_MAP = {
    # BPD grade -> (compliance_mean, compliance_std) in mL/cmH2O
    BPDGrade.NONE: (1.2, 0.15),
    BPDGrade.MILD: (0.85, 0.20),
    BPDGrade.MODERATE: (0.55, 0.18),
    BPDGrade.SEVERE: (0.30, 0.12),
}

BPD_GRADE_SURFACTANT_MAP = {
    BPDGrade.NONE: 0.80,
    BPDGrade.MILD: 0.65,
    BPDGrade.MODERATE: 0.45,
    BPDGrade.SEVERE: 0.25,
}


def load_patients(csv_path: str = "data/patients.csv") -> list[dict]:
    """
    Load patient data from CSV and initialize lung mechanics.

    Args:
        csv_path: Path to patients.csv

    Returns:
        List of dicts with 'profile' (PatientProfile) and 'lung_state' (LungState)
    """
    df = pd.read_csv(csv_path)
    patients = []
    
    for _, row in df.iterrows():
        bpd = BPDGrade(row["bpd_grade"])
        comp_mean, comp_std = BPD_GRADE_COMPLIANCE_MAP[bpd]

        # MRI-derived lung volume scales compliance
        # Healthy term baby lung volume ~150mL, premature ~30-80mL
        mri_volume = row.get("mri_lung_volume_ml", 60.0)
        volume_scaling = mri_volume / 60.0  # Normalise to median

        compliance = max(0.1, np.random.normal(comp_mean, comp_std) * volume_scaling)

        profile = PatientProfile(
            patient_id=str(row["patient_id"]),
            gestational_age_weeks=float(row["gestational_age_weeks"]),
            birth_weight_grams=float(row["birth_weight_grams"]),
            days_of_life=int(row["days_of_life"]),
            bpd_grade=bpd,
            lung_volume_ml=float(mri_volume),
            trachea_diameter_mm=float(row.get("trachea_diameter_mm", 6.0)),
        )

        lung_state = LungState(
            compliance=compliance,
            resistance=40.0 if bpd == BPDGrade.SEVERE else 25.0,
            frc=mri_volume * 0.35,
            surfactant=BPD_GRADE_SURFACTANT_MAP[bpd],
            time_step=0,
            patient_weight_kg=row["birth_weight_grams"] / 1000.0,
        )

        patients.append({"profile": profile, "lung_state": lung_state})

    return patients
