#!/usr/bin/env bash
# No network, SSH, scheduler submission, downloads or automatic retries.
set -euo pipefail
if [[ "$#" -ne 5 ]]; then
  echo 'Usage: bash run_cpu_server.sh REPO PYTHON DATA MODEL NEW_OUTPUT' >&2
  exit 2
fi
repo=$(realpath "$1")
py=$(realpath "$2")
data=$(realpath "$3")
model=$(realpath "$4")
out=$(realpath -m "$5")
task="$repo/research/gpu-prompt-boundary-handoff-20261012"
if [[ -e "$out" || -e "$out.server-logs" ]]; then
  echo 'Output or server logs already exist; choose a new run name' >&2
  exit 2
fi
# Match the frozen forward convention: 4 intra-op threads, 1 inter-op, one child.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
# handoff.py checks the output is outside actual git checkouts BEFORE this script
# creates logs. Preflight uses the same CPU build, manifests and scope checks.
"$py" - "$task" "$out" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from handoff import outside_git
outside_git(sys.argv[2])
PY
mkdir -p "$out.server-logs"
"$py" -m pip freeze > "$out.server-logs/environment.freeze.txt"
"$py" -m pip check > "$out.server-logs/pip-check.log" 2>&1
# Optional GNU time records real elapsed time and peak RSS on the user's server.
measure=()
if [[ -x /usr/bin/time ]]; then
  measure=(/usr/bin/time -v -o "$out.server-logs/run-resource.txt")
fi
"${measure[@]}" "$py" "$task/handoff.py" run --data "$data" --model "$model" \
  --out "$out" --device cpu > "$out.server-logs/run.stdout.log" 2> "$out.server-logs/run.stderr.log"
measure=()
if [[ -x /usr/bin/time ]]; then
  measure=(/usr/bin/time -v -o "$out.server-logs/validate-resource.txt")
fi
"${measure[@]}" "$py" "$task/handoff.py" validate --out "$out" \
  > "$out.server-logs/validate.stdout.log" 2> "$out.server-logs/validate.stderr.log"
echo "Validated CPU nine-trial batch; private results: $out"
