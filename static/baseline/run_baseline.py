"""
Baseline: runs GPT-4o against all 3 tasks and prints scores.
Usage:  OPENAI_API_KEY=sk-... python -m static.baseline.run_baseline
"""
import os
import json
import re
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package is required. Install with: pip install openai")
    sys.exit(1)

from models import NeoVentAction
from env.neovent_env import NeoVentEnv
from graders.grader import TaskGrader

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else None
env = NeoVentEnv()
grader = TaskGrader()

SYSTEM_PROMPT = """
You are an expert neonatal respiratory therapist managing mechanical ventilation
for a premature baby in the NICU. At each step you receive the baby's current
vitals and ventilator settings. You must decide small adjustments to keep the
baby stable while minimising lung damage.

Rules:
- Target SpO2: 91-95%. Below 88 is dangerous. Above 97 wastes oxygen.
- Target pCO2: 45-55 mmHg.
- Target Vt: 4-6 mL/kg body weight.
- Make SMALL changes. Never jump settings drastically.
- Wean FiO2 aggressively when SpO2 is in target.
- PEEP protects against lung collapse. Do not drop it below 4 without reason.

Respond ONLY with valid JSON matching this schema:
{
  "delta_pip":  <one of -2,-1,0,1,2>,
  "delta_peep": <one of -1,-0.5,0,0.5,1>,
  "delta_fio2": <one of -0.05,-0.02,0,0.02,0.05>,
  "delta_rr":   <one of -4,-2,0,2,4>,
  "reasoning":  "<brief clinical reasoning>"
}
"""


def _snap_delta(value: float, options: list[float]) -> float:
    return min(options, key=lambda x: abs(x - value))


def parse_action_message(raw_text: str | None) -> NeoVentAction | None:
    if not raw_text:
        return None

    candidates = [raw_text.strip()]
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return NeoVentAction(**payload)
        except Exception:
            continue
    return None


def heuristic_action(obs) -> NeoVentAction:
    weight_kg = max(obs.patient.birth_weight_grams / 1000.0, 0.4)
    vt_per_kg = obs.vitals.vt_ml / weight_kg
    spo2 = obs.vitals.spo2
    pco2 = obs.vitals.pco2
    fio2 = obs.current_settings.fio2
    peep = obs.current_settings.peep

    delta_pip = 0.0
    delta_peep = 0.0
    delta_fio2 = 0.0
    delta_rr = 0.0

    if spo2 < 82:
        delta_fio2 = 0.05
        delta_pip = 2.0
        delta_peep = 1.0
    elif spo2 < 88:
        delta_fio2 = 0.05
        delta_pip = 1.0
        delta_peep = 0.5
    elif spo2 < 91:
        delta_fio2 = 0.02
        if vt_per_kg < 4.0:
            delta_pip = 1.0
    elif spo2 > 97:
        delta_fio2 = -0.05
        if vt_per_kg > 6.5:
            delta_pip = -1.0
    elif spo2 > 95:
        delta_fio2 = -0.02

    if pco2 > 65:
        delta_rr = 4.0
    elif pco2 > 58:
        delta_rr = 2.0
    elif pco2 < 30 and spo2 > 92:
        delta_rr = -4.0
    elif pco2 < 38 and spo2 > 92:
        delta_rr = -2.0

    if spo2 >= 90:
        if vt_per_kg > 9.0:
            delta_pip = min(delta_pip, -2.0)
        elif vt_per_kg > 7.5:
            delta_pip = min(delta_pip, -1.0)
        elif vt_per_kg < 3.5 and spo2 < 92:
            delta_pip = max(delta_pip, 1.0)

    if spo2 < 90 and peep < 6.5:
        delta_peep = max(delta_peep, 0.5)
    if spo2 > 95 and fio2 <= 0.25 and peep > 5.0:
        delta_peep = min(delta_peep, -0.5)

    return NeoVentAction(
        delta_pip=_snap_delta(delta_pip, [-2, -1, 0, 1, 2]),
        delta_peep=_snap_delta(delta_peep, [-1, -0.5, 0, 0.5, 1]),
        delta_fio2=_snap_delta(delta_fio2, [-0.05, -0.02, 0, 0.02, 0.05]),
        delta_rr=int(_snap_delta(delta_rr, [-4, -2, 0, 2, 4])),
        reasoning="Heuristic rescue policy",
    )


def run_task(task_id: str) -> float:
    """Run a single task with GPT-4o baseline."""
    obs = env.reset(task_id)
    assert env.simulator is not None
    history = []
    done = False

    print(f"\n{'='*60}")
    print(f"Running {task_id}...")
    print(f"{'='*60}")

    while not done:
        user_msg = f"""
Patient: {obs.patient.gestational_age_weeks}wk gestation,
         {obs.patient.birth_weight_grams}g, BPD={obs.patient.bpd_grade.value}
Vitals:  SpO2={obs.vitals.spo2}%, pCO2={obs.vitals.pco2}mmHg,
         HR={obs.vitals.hr}bpm, Vt={obs.vitals.vt_ml}mL
Settings: PIP={obs.current_settings.pip}, PEEP={obs.current_settings.peep},
          FiO2={obs.current_settings.fio2:.2f}, RR={obs.current_settings.rr}
Alarms:  {obs.alarm_flags}
Step:    {obs.step_number}
Barotrauma index: {obs.cumulative_barotrauma_index:.3f}
"""
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )

                raw = response.choices[0].message.content
                action = parse_action_message(raw)
                if action is None:
                    print("  Invalid model JSON. Using heuristic action.")
                    action = heuristic_action(obs)
            except Exception as e:
                print(f"  Model error: {e}. Using heuristic action.")
                action = heuristic_action(obs)
        else:
            action = heuristic_action(obs)

        history.append({
            "vitals": obs.vitals,
            "settings": obs.current_settings,
            "initial_compliance": env.simulator.state.compliance if len(history) == 0 else history[0].get("initial_compliance"),
        })

        obs, reward_dict, done, info = env.step(action)
        print(f"  Step {info['step']:3d} | SpO2={obs.vitals.spo2:5.1f}% "
              f"| FiO2={obs.current_settings.fio2:.2f} "
              f"| Reward={reward_dict['total']:+.2f}")

    score = grader.grade(env, history)
    return score


def main():
    """Run baseline on all tasks."""
    if client is None:
        print("WARN: OPENAI_API_KEY not set. Running deterministic heuristic baseline.")

    results = {}
    for task in ["task_easy", "task_medium", "task_hard"]:
        try:
            score = run_task(task)
            results[task] = score
            print(f"Score: {score:.3f}")
        except Exception as e:
            print(f"ERROR in {task}: {e}")
            results[task] = 0.0

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"  Average: {avg:.3f}")


if __name__ == "__main__":
    main()
