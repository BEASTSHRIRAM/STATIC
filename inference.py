#!/usr/bin/env python3
"""
NeoVentEnv Inference Script

Runs GPT-4o baseline agent against NeoVentEnv tasks.
Follows the OpenEnv submission format with structured logging.

Environment variables:
  API_BASE_URL  - OpenAI API base URL (default: https://api.openai.com/v1)
  API_KEY       - OpenAI API key (required)
  MODEL_NAME    - Model identifier (default: gpt-4o)
"""

import asyncio
import json
import os
import re
import sys
from typing import List, Optional
from datetime import datetime

try:
    from openai import OpenAI
    from static import NeoVentAction
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}", flush=True)
    sys.exit(1)


# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Support running from repository root where env modules expect the static/ dir on sys.path.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

# Submission parameters
BENCHMARK = "NeoVentEnv"
IMAGE_NAME = "neovent:latest"
MAX_STEPS = 150
SUCCESS_SCORE_THRESHOLD = 0.50
TASKS = ["task_easy", "task_medium", "task_hard"]


def _snap_delta(value: float, options: list[float]) -> float:
    """Snap a continuous value to the nearest allowed discrete delta."""
    return min(options, key=lambda x: abs(x - value))


def parse_action_message(raw_text: str) -> Optional[NeoVentAction]:
    """Parse model text into NeoVentAction, supporting fenced or mixed content."""
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


def heuristic_action(obs_dict: dict) -> NeoVentAction:
    """Deterministic rescue policy used when model output is unavailable or invalid."""
    vitals = obs_dict.get("vitals", {})
    settings = obs_dict.get("current_settings", {})
    patient = obs_dict.get("patient", {})

    weight_kg = max(float(patient.get("birth_weight_grams", 1000)) / 1000.0, 0.4)
    spo2 = float(vitals.get("spo2", 92.0))
    pco2 = float(vitals.get("pco2", 50.0))
    vt_ml = float(vitals.get("vt_ml", 5.0))
    vt_per_kg = vt_ml / weight_kg
    fio2 = float(settings.get("fio2", 0.4))
    peep = float(settings.get("peep", 5.0))

    delta_pip = 0.0
    delta_peep = 0.0
    delta_fio2 = 0.0
    delta_rr = 0.0

    # Oxygenation first: prevent critical hypoxia quickly.
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

    # CO2 control via rate tuning.
    if pco2 > 65:
        delta_rr = 4.0
    elif pco2 > 58:
        delta_rr = 2.0
    elif pco2 < 30 and spo2 > 92:
        delta_rr = -4.0
    elif pco2 < 38 and spo2 > 92:
        delta_rr = -2.0

    # Lung-protective volume adjustments.
    if spo2 >= 90:
        if vt_per_kg > 9.0:
            delta_pip = min(delta_pip, -2.0)
        elif vt_per_kg > 7.5:
            delta_pip = min(delta_pip, -1.0)
        elif vt_per_kg < 3.5 and spo2 < 92:
            delta_pip = max(delta_pip, 1.0)

    # PEEP guardrails.
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


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in OpenEnv format."""
    print(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "start",
            "task": task,
            "env": env,
            "model": model,
        }),
        flush=True
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Log step result in OpenEnv format."""
    print(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "step",
            "step": step,
            "action": action[:100] if isinstance(action, str) else str(action),
            "reward": round(reward, 4),
            "done": done,
            "error": error,
        }),
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in OpenEnv format."""
    print(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "end",
            "success": success,
            "steps": steps,
            "score": round(score, 4),
            "total_reward": round(sum(rewards), 4),
            "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        }),
        flush=True
    )


def get_model_message(
    client: OpenAI,
    step: int,
    obs_dict: dict,
    last_reward: float,
    history: List[str],
) -> str:
    """
    Query GPT-4o for next action.
    
    Args:
        client: OpenAI client
        step: Current step number
        obs_dict: Current observation as dict
        last_reward: Last step reward
        history: Episode history
        
    Returns:
        Model's response (action string)
    """
    system_prompt = """
You are an expert neonatal respiratory therapist managing mechanical ventilation for a premature baby in the NICU.

At each step you receive the baby's vitals and ventilator settings. You must decide small adjustments to keep the baby stable while minimising lung damage.

OBJECTIVES:
1. Keep SpO2 in target range 91-95% (critical if < 75%)
2. Minimize tidal volume (Vt) per kg body weight, target 4-6 mL/kg
3. Avoid unnecessary oxygen (FiO2 weaning)
4. Protect lungs from barotrauma (volutrauma + atelecttrauma)

RULES:
- Make SMALL changes. Valid deltas: PIP {-2,-1,0,1,2}, PEEP {-1,-0.5,0,0.5,1}, FiO2 {-0.05,-0.02,0,0.02,0.05}, RR {-4,-2,0,2,4}
- PEEP protects against lung collapse. Never drop below 4 unless necessary.
- Wean FiO2 aggressively when SpO2 is in target range.
- If SpO2 < 88%, increase PIP and FiO2 immediately.
- If pCO2 > 60 (hypercapnia), increase RR.
- If pCO2 < 35 (hypocapnia), decrease RR.

Respond ONLY with valid JSON matching this schema:
{
  "delta_pip": <-2,-1,0,1,2>,
  "delta_peep": <-1,-0.5,0,0.5,1>,
  "delta_fio2": <-0.05,-0.02,0,0.02,0.05>,
  "delta_rr": <-4,-2,0,2,4>,
  "reasoning": "<brief clinical reasoning>"
}
"""

    vitals = obs_dict.get("vitals", {})
    settings = obs_dict.get("current_settings", {})
    patient = obs_dict.get("patient", {})
    
    weight_kg = patient.get("birth_weight_grams", 1000) / 1000.0
    vt_per_kg = vitals.get("vt_ml", 0) / weight_kg if weight_kg > 0 else 0
    
    user_msg = f"""
PATIENT PROFILE:
- {patient.get('gestational_age_weeks', '?')}w gestation, {patient.get('birth_weight_grams', '?')}g
- BPD Grade: {patient.get('bpd_grade', 'unknown')}
- Days of life: {patient.get('days_of_life', '?')}

CURRENT VITALS:
- SpO2: {vitals.get('spo2', '?')}% (target 91-95%, critical < 75%)
- pCO2: {vitals.get('pco2', '?')} mmHg (target 45-55)
- HR: {vitals.get('hr', '?')} bpm (normal 120-160)
- Vt: {vitals.get('vt_ml', '?')} mL ({vt_per_kg:.2f} mL/kg, target 4-6)

CURRENT SETTINGS:
- PIP: {settings.get('pip', '?')} cmH2O (range 10-35)
- PEEP: {settings.get('peep', '?')} cmH2O (range 2-10)
- FiO2: {settings.get('fio2', '?'):.2f} (range 0.21-1.0)
- RR: {settings.get('rr', '?')} breaths/min (range 15-80)
- Mode: {settings.get('mode', 'SIMV')}

STEP: {step} / {MAX_STEPS}
LAST REWARD: {last_reward:+.2f}
BAROTRAUMA INDEX: {obs_dict.get('cumulative_barotrauma_index', 0):.3f}
ALARMS: {', '.join(obs_dict.get('alarm_flags', []))}

Adjust ONE or TWO parameters. Explain your reasoning briefly.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return ""


async def run_task(task_id: str, client: Optional[OpenAI]) -> dict:
    """
    Run a single task with the baseline agent.
    
    Args:
        task_id: One of task_easy, task_medium, task_hard
        client: OpenAI client
        
    Returns:
        dict with score, steps, rewards
    """
    try:
        # In production: connect to running HF Space or docker image
        # For now, use local environment
        from env.neovent_env import NeoVentEnv
        from graders.grader import TaskGrader
        
        env = NeoVentEnv(data_path=os.path.join(STATIC_DIR, "data", "patients.csv"))
        grader = TaskGrader()
        episode_history: list[dict] = []
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            # Reset environment
            obs = env.reset(task_id)
            assert env.simulator is not None
            obs_dict = obs.model_dump()
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                episode_history.append({
                    "vitals": obs.vitals,
                    "settings": obs.current_settings,
                    "initial_compliance": (
                        env.simulator.state.compliance
                        if not episode_history
                        else episode_history[0].get("initial_compliance")
                    ),
                })

                # Get action from model
                if client is not None:
                    action_str = get_model_message(client, step, obs_dict, last_reward, history)
                    action = parse_action_message(action_str)
                    if action is None:
                        print("[DEBUG] Invalid model action; using heuristic fallback", flush=True)
                        action = heuristic_action(obs_dict)
                        action_str = json.dumps(action.model_dump())
                else:
                    action = heuristic_action(obs_dict)
                    action_str = json.dumps(action.model_dump())
                
                # Step environment
                obs, reward_dict, done, info = env.step(action)
                
                reward = reward_dict.get("total", 0.0) if isinstance(reward_dict, dict) else float(reward_dict)
                obs_dict = obs.model_dump()
                
                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                
                log_step(
                    step=step,
                    action=action_str[:50],
                    reward=reward,
                    done=done,
                    error=None
                )
                
                history.append(f"Step {step}: reward={reward:+.2f}, SpO2={obs.vitals.spo2:.1f}%")
                
                if done:
                    break

            # Compute score using the same grader as task evaluation.
            score = grader.grade(env, episode_history)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            try:
                pass  # env.close() if using client
            except Exception as e:
                print(f"[DEBUG] close() error: {e}", flush=True)
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        return {
            "task": task_id,
            "score": score,
            "steps": steps_taken,
            "rewards": rewards,
            "success": success,
        }

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {
            "task": task_id,
            "score": 0.0,
            "steps": 0,
            "rewards": [],
            "success": False,
        }


async def main() -> None:
    """Run baseline on all tasks."""
    if API_KEY:
        client: Optional[OpenAI] = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    else:
        client = None
        print("[WARN] OPENAI_API_KEY not set; using deterministic heuristic baseline", flush=True)
    results = {}

    print("="*60, flush=True)
    print(f"NeoVentEnv Baseline: {MODEL_NAME}", flush=True)
    print("="*60, flush=True)

    for task in TASKS:
        try:
            result = await run_task(task, client)
            results[task] = result
        except Exception as e:
            print(f"[ERROR] {task}: {e}", flush=True)
            results[task] = {"task": task, "score": 0.0, "steps": 0, "rewards": [], "success": False}

    # Summary
    print("\n" + "="*60, flush=True)
    print("BASELINE RESULTS", flush=True)
    print("="*60, flush=True)
    
    total_score = 0.0
    for task in TASKS:
        result = results[task]
        score = result.get("score", 0.0)
        total_score += score
        print(f"{task}: {score:.3f}", flush=True)
    
    avg_score = total_score / len(TASKS) if TASKS else 0.0
    print(f"Average: {avg_score:.3f}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
