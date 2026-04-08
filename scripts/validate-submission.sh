#!/usr/bin/env bash
#
# validate-neovent-submission.sh — NeoVentEnv Submission Validator
#
# Checks that NeoVentEnv environment meets OpenEnv submission requirements
#

set -uo pipefail

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

REPO_DIR="${1:-.}"
PASSED=0
FAILED=0

check_pass() {
  echo -e "${GREEN}✓ $1${NC}" >&2
  ((PASSED++))
}

check_fail() {
  echo -e "${RED}✗ $1${NC}" >&2
  ((FAILED++))
}

check_warn() {
  echo -e "${YELLOW}⚠ $1${NC}" >&2
}

echo -e "${BOLD}NeoVentEnv Submission Validator${NC}" >&2
echo "======================================" >&2

# Check 1: openenv.yaml exists
if [ -f "$REPO_DIR/static/openenv.yaml" ]; then
  check_pass "openenv.yaml exists"
else
  check_fail "openenv.yaml not found"
fi

# Check 2: Required Python files
for file in \
  "static/models.py" \
  "static/env/neovent_env.py" \
  "static/env/lung_simulator.py" \
  "static/env/patient_loader.py" \
  "static/graders/grader.py" \
  "static/baseline/run_baseline.py" \
  "static/tests/test_env.py" \
  "static/server/app.py" \
  "static/server/static_environment.py"
do
  if [ -f "$REPO_DIR/$file" ]; then
    check_pass "Found $file"
  else
    check_fail "Missing $file"
  fi
done

# Check 3: Data files
if [ -f "$REPO_DIR/static/data/patients.csv" ]; then
  check_pass "Patient data (patients.csv) exists"
  LINES=$(wc -l < "$REPO_DIR/static/data/patients.csv")
  if [ "$LINES" -ge 40 ]; then
    check_pass "  Contains $LINES patient records (>= 40 required)"
  else
    check_warn "  Only $LINES records (40+ recommended)"
  fi
else
  check_fail "Patient data missing"
fi

# Check 4: Dockerfile
if [ -f "$REPO_DIR/static/server/Dockerfile" ]; then
  check_pass "Dockerfile exists"
  if grep -q "python:3" "$REPO_DIR/static/server/Dockerfile"; then
    check_pass "  Uses Python base image"
  fi
  if grep -q "requirements.txt" "$REPO_DIR/static/server/Dockerfile"; then
    check_pass "  Installs requirements"
  fi
else
  check_fail "Dockerfile not found"
fi

# Check 5: requirements.txt
if [ -f "$REPO_DIR/static/server/requirements.txt" ]; then
  check_pass "requirements.txt exists"
  for pkg in "openenv" "fastapi" "uvicorn" "pydantic" "numpy" "pandas"; do
    if grep -q "$pkg" "$REPO_DIR/static/server/requirements.txt"; then
      check_pass "  Contains $pkg"
    else
      check_warn "  Missing $pkg"
    fi
  done
else
  check_fail "requirements.txt not found"
fi

# Check 6: README
if [ -f "$REPO_DIR/static/README.md" ]; then
  check_pass "README.md exists"
  README="$REPO_DIR/static/README.md"
  
  for section in "Installation" "Usage" "Tasks" "Observation" "Action" "Reward"; do
    if grep -qi "$section" "$README"; then
      check_pass "  Includes $section section"
    else
      check_warn "  Missing $section section"
    fi
  done
else
  check_fail "README.md not found"
fi

# Check 7: inference.py
if [ -f "$REPO_DIR/inference.py" ]; then
  check_pass "inference.py exists in root"
  if grep -q "log_start\|log_step\|log_end" "$REPO_DIR/inference.py"; then
    check_pass "  Contains required log functions"
  else
    check_warn "  Missing log functions"
  fi
else
  check_warn "inference.py not found (optional)"
fi

# Check 8: Models typing
if grep -q "class NeoVentAction" "$REPO_DIR/static/models.py" \
   && grep -q "class NeoVentObservation" "$REPO_DIR/static/models.py"; then
  check_pass "Pydantic models defined (NeoVentAction, NeoVentObservation)"
else
  check_fail "Pydantic models not found or incorrectly named"
fi

# Check 9: Tasks defined
if grep -q "task_easy\|task_medium\|task_hard" "$REPO_DIR/static/env/neovent_env.py"; then
  check_pass "3 tasks defined (easy, medium, hard)"
else
  check_fail "Tasks not defined"
fi

# Check 10: Grader implemented
if grep -q "class TaskGrader" "$REPO_DIR/static/graders/grader.py" \
   && grep -q "def grade" "$REPO_DIR/static/graders/grader.py"; then
  check_pass "TaskGrader class with grade() method"
else
  check_fail "TaskGrader not found"
fi

echo ""
echo "======================================" >&2
echo -e "${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}" >&2
echo "======================================" >&2

if [ $FAILED -eq 0 ]; then
  exit 0
else
  exit 1
fi
