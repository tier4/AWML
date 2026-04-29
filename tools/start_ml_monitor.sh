#!/bin/bash
# Start MLflow (port 5500) and TensorBoard (port 6600) inside the autoware-ml-integration
# container as separate panels in a tmux session named ml_monitor.
#
# Usage: bash tools/start_ml_monitor.sh [work_dir_inside_container]
#   work_dir defaults to /workspace/AWML
#
# Run from the host (not inside the container).

set -e

CONTAINER="autoware-ml-integration"
WORKSPACE="${1:-/workspace/AWML}"
SESSION="ml_monitor"
MLFLOW_PORT=5500
TB_PORT=6600

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: container '${CONTAINER}' is not running."
    echo "Start it with:  docker start ${CONTAINER}"
    exit 1
fi

# Check for port conflicts on host
for port in ${MLFLOW_PORT} ${TB_PORT}; do
    if ss -tlnp 2>/dev/null | grep -q ":${port} " || \
       netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "WARNING: host port ${port} already in use — server on that port will not start."
    fi
done

# Create or attach to session inside the container via docker exec
# We launch tmux inside the container so the servers run there.
docker exec -it "${CONTAINER}" bash -c "
    tmux has-session -t ${SESSION} 2>/dev/null && {
        echo 'Session ${SESSION} already exists — attaching.';
        tmux attach -t ${SESSION};
        exit 0;
    }

    # New session: window 0 = MLflow
    tmux new-session -d -s ${SESSION} -n mlflow \
        \"mlflow server --host 0.0.0.0 --port ${MLFLOW_PORT} \
            --backend-store-uri ${WORKSPACE}/mlruns 2>&1 | tee /tmp/mlflow.log\"

    # Window 1 = TensorBoard
    tmux new-window -t ${SESSION}:1 -n tensorboard \
        \"tensorboard --logdir ${WORKSPACE}/work_dirs \
            --host 0.0.0.0 --port ${TB_PORT} 2>&1 | tee /tmp/tensorboard.log\"

    echo ''
    echo 'ml_monitor session started inside container ${CONTAINER}:'
    echo \"  MLflow    → http://localhost:${MLFLOW_PORT}\"
    echo \"  TensorBoard → http://localhost:${TB_PORT}\"
    echo ''
    echo 'To attach:  docker exec -it ${CONTAINER} tmux attach -t ${SESSION}'
    echo 'To switch panels inside tmux: Ctrl-b 0 (mlflow) / Ctrl-b 1 (tensorboard)'
"
