#!/bin/bash
# Stage 11 step 4: wraps train_residual.py with automatic retry-from-checkpoint.
# Exists specifically because of an observed intermittent crash in this controller's own
# native libraries (CasADi/Pinocchio/OSQP) under this sandbox's load -- see mpc/README.md.
# Checkpoints save every iteration (see train_residual.py's save_freq comment), so a crash
# loses at most one iteration's worth of progress, not the whole run.
#
# Usage: bash mpc/run_resilient.sh --run-name v1 --timesteps 1000000 --n-envs 8 ...
set -u
cd "$(dirname "$0")"
source venv/bin/activate

MAX_ATTEMPTS=30
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== attempt $attempt/$MAX_ATTEMPTS: $(date) ==="
    python3 train_residual.py --auto-resume "$@"
    code=$?
    if [ $code -eq 0 ]; then
        echo "=== exited cleanly (code 0) on attempt $attempt ==="
        exit 0
    fi
    echo "=== crashed (exit code $code) on attempt $attempt, retrying from last checkpoint ==="
    sleep 5
done
echo "=== gave up after $MAX_ATTEMPTS attempts ==="
exit 1
