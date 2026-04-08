---
title: NeoVentEnv - Neonatal Ventilator Management
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - reinforcement-learning
  - neonatal
---

# NeoVentEnv

NeoVentEnv is an OpenEnv-compatible simulation where an agent controls ventilator settings for premature neonates in a NICU-like scenario.

At each step, the agent observes current clinical signals and chooses small, safe ventilator adjustments. The objective is to keep oxygenation in target range while reducing lung injury and oxygen toxicity.

## Why This Environment Exists

Bronchopulmonary dysplasia (BPD) risk increases when ventilation is poorly managed. In this environment, the agent must balance:

- oxygenation stability (SpO2 target 91-95)
- carbon dioxide control (pCO2 target 45-55)
- lung-protective tidal volume (4-6 mL/kg)
- reduced unnecessary oxygen exposure

This is intentionally a tradeoff problem, not a single-metric optimization.

## Tasks

NeoVentEnv provides three benchmark tasks:

| Task ID | Difficulty | Max Steps | Typical Patient Profile |
|---|---|---:|---|
| `task_easy` | easy | 40 | stable, none/mild BPD |
| `task_medium` | medium | 80 | moderate drift, mild/moderate BPD |
| `task_hard` | hard | 120 | unstable mechanics, moderate/severe BPD |

Task metadata is defined in `openenv.yaml`.

## Observation Space

Observation model: `NeoVentObservation`.

Main fields:

- `patient`: static profile (gestational age, weight, BPD grade, lung volume)
- `vitals`: dynamic clinical values (`spo2`, `pco2`, `hr`, `vt_ml`)
- `current_settings`: ventilator settings (`pip`, `peep`, `fio2`, `rr`, `mode`)
- `step_number`
- `time_on_vent_hrs`
- `cumulative_barotrauma_index`
- `alarm_flags`
- `context`

## Action Space

Action model: `NeoVentAction`.

Valid deltas:

- `delta_pip`: `{-2, -1, 0, 1, 2}`
- `delta_peep`: `{-1, -0.5, 0, 0.5, 1}`
- `delta_fio2`: `{-0.05, -0.02, 0, 0.02, 0.05}`
- `delta_rr`: `{-4, -2, 0, 2, 4}`

Invalid values are snapped to nearest valid deltas by `_validate_action` in `env/neovent_env.py`.

## Reward Function

Reward is shaped at every step using multiple terms:

- SpO2 targeting score
- hypoxia penalty (strong cliff below severe thresholds)
- gentleness score via tidal volume per kg
- hyperoxia cost when oxygen is unnecessarily high
- barotrauma accumulation cost

Total reward is the sum of these components and is returned in `reward_dict["total"]`.

## Termination Conditions

An episode can end due to:

- maximum step count reached for the task
- critical hypoxia
- severe barotrauma threshold

Termination reason is tracked in environment state.

## Project Structure

```text
static/
|- openenv.yaml
|- models.py
|- env/
|  |- neovent_env.py
|  |- lung_simulator.py
|  |- patient_loader.py
|- graders/
|  |- grader.py
|- baseline/
|  |- run_baseline.py
|- server/
|  |- app.py
|  |- static_environment.py
|  |- requirements.txt
|  |- Dockerfile
|- tests/
|  |- test_env.py
|- data/
|  |- patients.csv
```

## Local Setup

From repository root:

```bash
cd /workspaces/STATIC
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r static/server/requirements.txt
```

## Running Tests

```bash
cd /workspaces/STATIC/static
pytest tests/test_env.py -v
```

## Running the Baseline

```bash
cd /workspaces/STATIC/static
export OPENAI_API_KEY=your_key_here
python -m baseline.run_baseline
```

If `OPENAI_API_KEY` is missing, the baseline falls back to deterministic heuristic actions.

## Starting the API Server

```bash
cd /workspaces/STATIC/static
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Useful routes:

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /docs`
- `GET /web/`

## API Examples

Set base URL:

```bash
BASE=http://localhost:8000
```

Reset task:

```bash
curl -s -X POST "$BASE/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_easy"}'
```

Step once (note the `action` envelope):

```bash
curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "delta_pip": 0,
      "delta_peep": 0,
      "delta_fio2": -0.02,
      "delta_rr": 0,
      "reasoning": "gentle wean in stable oxygen range"
    }
  }'
```

Get current state:

```bash
curl -s "$BASE/state"
```

## Manual Playground Test Presets

Use these values in `/docs` or `/web/` for quick sanity checks.

1. Stable weaning test

- `delta_pip: 0`
- `delta_peep: 0`
- `delta_fio2: -0.02`
- `delta_rr: 0`

Expected: if SpO2 remains in range, FiO2 gradually decreases.

2. Rescue low SpO2

- `delta_pip: 1`
- `delta_peep: 0.5`
- `delta_fio2: 0.02`
- `delta_rr: 2`

Expected: oxygenation should recover without extreme overshoot.

3. Hyperoxia correction

- `delta_pip: 0`
- `delta_peep: 0`
- `delta_fio2: -0.05`
- `delta_rr: 0`

Expected: SpO2 drifts down from high values and oxygen exposure improves.

4. Hypercapnia correction (`pCO2 > 60`)

- `delta_pip: 0`
- `delta_peep: 0`
- `delta_fio2: 0`
- `delta_rr: 2` or `4`

Expected: CO2 clears over subsequent steps.

5. Hypocapnia correction (`pCO2 < 35`)

- `delta_pip: 0`
- `delta_peep: 0`
- `delta_fio2: 0`
- `delta_rr: -2` or `-4`

Expected: pCO2 normalizes upward.

## Grading

`graders/grader.py` returns a deterministic score in `[0.0, 1.0]` based on:

- time in SpO2 target range
- critical hypoxia events
- compliance preservation (lung protection)
- average FiO2 (oxygen weaning quality)
- termination penalties

Expected baseline ranges (from `openenv.yaml`):

- `task_easy`: `0.65-0.75`
- `task_medium`: `0.40-0.55`
- `task_hard`: `0.20-0.35`

## Submission Validation

From repo root:

```bash
cd /workspaces/STATIC
bash scripts/validate-submission.sh
```

## Deployment

Push this environment to a Hugging Face Space using OpenEnv:

```bash
cd /workspaces/STATIC
source .venv/bin/activate
openenv push static --repo-id Beast7878/neovent
```

## Troubleshooting

1. `Call reset() before step()`

- Ensure you call `/reset` first.
- Ensure `/step` request uses `{"action": {...}}` wrapper.

2. Invalid action format

- Use exact field names: `delta_pip`, `delta_peep`, `delta_fio2`, `delta_rr`.

3. Import errors in server container

- Verify `server/__init__.py` exports `NeoVentEnvironment` and compatibility alias.

4. Low score

- Check for no-op actions, overly conservative settings, or delayed hypoxia response.
- Use rescue preset for low SpO2 and then transition back to weaning.

## Safety Note

This environment is for simulation and benchmarking only. It is not a medical device and is not intended for direct clinical decision making.