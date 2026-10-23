source /m2v_intern2/zhenglin/my_bashrc.sh
set -euo pipefail
source "${H3_ATTEMPT_DIR}/launch.env"
source "${H3_CHECKOUT}/scripts/minimax_h3_runtime_env.sh"
export H3_MODEL="${H3_MODEL:?missing prepared local model}"
export H3_REWARD_MANIFEST="${H3_REWARD_MANIFEST:?missing seven-asset reward manifest}"
export H3_TRAIN_MANIFEST="${H3_TRAIN_MANIFEST:?missing fixed training dataset}"
export H3_TRAIN_SHA256="${H3_TRAIN_SHA256:?missing dataset SHA256}"
export H3_TRACE_BATCH_SHAPES="${H3_TRACE_BATCH_SHAPES:-0}"
export H3_RUNTIME_COMMIT
export TRAINING_COMMIT="${H3_RUNTIME_COMMIT}"
export VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="${H3_COLLECTIVE_TIMEOUT_SECONDS:-1800}"
extra=()
if [[ -n "${H3_RESUME:-}" ]]; then
  extra+=(--resume "${H3_RESUME}")
fi
if [[ "${H3_RESUME_REWIND:-0}" == 1 ]]; then
  extra+=(--resume-rewind)
fi
if [[ "${H3_EVAL_ONLY:-0}" == 1 ]]; then
  extra+=(--eval-only)
fi
printf 'H3 native train node=%s source=%s config=%s preparation=%s\n' \
  "${H3_NODE_RANK}" "${H3_RUNTIME_COMMIT}" "${H3_ACCEPTANCE_CONFIG}" "${H3_PREPARATION_RUN_ID}"
exec "${H3_PYTHON}" -m visual_rl.train \
  --config "${H3_ACCEPTANCE_CONFIG}" --nproc-per-node 8 \
  --nnodes "${H3_NNODES}" --node-rank "${H3_NODE_RANK}" \
  --master-addr "${H3_MASTER_ADDR}" --master-port "${H3_MASTER_PORT}" \
  --preparation-run-id "${H3_PREPARATION_RUN_ID}" "${extra[@]}"
