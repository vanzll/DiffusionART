source /m2v_intern2/zhenglin/my_bashrc.sh
set -euo pipefail
source "${H3_ATTEMPT_DIR}/launch.env"
source "${H3_CHECKOUT}/scripts/minimax_h3_runtime_env.sh"
exec "${H3_PYTHON}" -m torch.distributed.run \
  --nnodes "${H3_NNODES}" --node-rank "${H3_NODE_RANK}" --nproc-per-node 8 \
  --master-addr "${H3_MASTER_ADDR}" --master-port "${H3_MASTER_PORT}" \
  --log-dir "${H3_ATTEMPT_DIR}/logs/torchrun" --redirects 3 --tee 3 \
  "${H3_CHECKOUT}/scripts/minimax_h3_distributed_gate.py" \
  --layout "${H3_GATE_LAYOUT:?missing actual H3 block layout receipt}"
