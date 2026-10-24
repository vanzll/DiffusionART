source /m2v_intern2/zhenglin/my_bashrc.sh
set -euo pipefail
unset H3_ATTEMPT_NONCE
source "${1:?missing attempt environment}"
export PYTHONDONTWRITEBYTECODE=1
node_rank="${OMPI_COMM_WORLD_RANK:?missing MPI node rank}"
host="$(hostname -s)"
control="${H3_CHECKOUT}/scripts/minimax_h3_process_control.py"
node_dir="${H3_ATTEMPT_DIR}/nodes/${node_rank}"
mkdir -p "${node_dir}" "${H3_ATTEMPT_DIR}/logs"
exec > "${node_dir}/owner.log" 2>&1
source "${H3_LEASE_LIBRARY:?missing existing fenced lease capability}"
child_pid=""
stop_requested=0
owner_start="$(flowart_pid_start_ticks $$)"

owner_event() {
  printf '%s host=%s node_rank=%s owner_pid=%s %s\n' "$(date -u +%FT%TZ)" "${host}" "${node_rank}" "$$" "$*"
}

stop_signal() {
  stop_requested=1
  printf 'signal\n' > "${H3_ATTEMPT_DIR}/failure.${node_rank}"
}

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if (( status != 0 )); then
    printf 'exit=%s\n' "${status}" > "${H3_ATTEMPT_DIR}/failure.${node_rank}"
  fi
  if [[ "${FLOWART_GPU_LEASE_OWNED:-0}" == 1 ]]; then
    owner_event "draining nonce=${H3_ATTEMPT_NONCE} generation=${FLOWART_GPU_LEASE_GENERATION}"
    if ! timeout -k 5 100 /usr/bin/python3 "${control}" \
      --lease "${FLOWART_GPU_LEASE_OWNER}" --owner-pid "$$" \
      --owner-start-ticks "${owner_start}" --generation "${FLOWART_GPU_LEASE_GENERATION}" \
      --nonce "${H3_ATTEMPT_NONCE}" --root "${H3_ACCEPTANCE_ROOT}" --terminate \
      > "${node_dir}/drain.json" 2> "${node_dir}/drain.stderr"; then
      flowart_gpu_lease_quarantine "${child_pid:-$$}" nonce-drain-unproven || true
      owner_event 'QUARANTINED: descendant drain unproven'
      printf 'drain-failed\n' > "${H3_ATTEMPT_DIR}/failure.${node_rank}"
      printf '124\n' > "${node_dir}/exitcode"
      exit 124
    fi
    if (( status == 0 )) && ! /usr/bin/python3 -c \
      'import json,sys; result=json.load(open(sys.argv[1])); sys.exit(bool(result["signaled"]))' \
      "${node_dir}/drain.json"; then
      status=125
      owner_event 'leader reported success but left descendants; attempt failed'
      printf 'orphan-descendants\n' > "${H3_ATTEMPT_DIR}/failure.${node_rank}"
    fi
    local deadline=$((SECONDS + 90))
    until timeout -k 2 12 /usr/bin/python3 "${control}" --empty > "${node_dir}/gpu-drain.json" 2>&1; do
      if (( SECONDS >= deadline )); then
        flowart_gpu_lease_quarantine "${child_pid:-$$}" gpu-context-drain-unproven || true
        owner_event 'QUARANTINED: GPU context drain unproven'
        printf 'gpu-drain-failed\n' > "${H3_ATTEMPT_DIR}/failure.${node_rank}"
        printf '124\n' > "${node_dir}/exitcode"
        exit 124
      fi
      sleep 2
    done
    [[ -z "${child_pid}" ]] || wait "${child_pid}" 2>/dev/null || true
    flowart_gpu_lease_release
    owner_event 'drained; original fenced holder restoration requested'
  fi
  printf '%s\n' "${status}" > "${node_dir}/exitcode"
  exit "${status}"
}

trap cleanup EXIT
trap stop_signal TERM INT
owner_event "admission nonce=${H3_ATTEMPT_NONCE}"
[[ "$(git -C "${H3_CHECKOUT}" rev-parse HEAD)" == "${H3_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${H3_CHECKOUT}" status --porcelain)" ]]
timeout -k 2 12 /usr/bin/python3 "${control}" --admit > "${node_dir}/admission.json"
flowart_gpu_lease_configure "${host}" h3-art-full-acceptance "${H3_ATTEMPT_NONCE}" "${H3_ATTEMPT_NONCE}"
flowart_gpu_lease_acquire
owner_event "acquired generation=${FLOWART_GPU_LEASE_GENERATION}"
printf 'pid=%s\nstart_ticks=%s\ngeneration=%s\nhost=%s\n' "$$" "${owner_start}" "${FLOWART_GPU_LEASE_GENERATION}" "${host}" > "${node_dir}/owner.identity"
deadline=$((SECONDS + 120))
until timeout -k 2 12 /usr/bin/python3 "${control}" --empty > "${node_dir}/post-free.json" 2>&1; do
  (( SECONDS < deadline && stop_requested == 0 )) || exit 125
  sleep 2
done
printf 'ready\n' > "${node_dir}/ready"
deadline=$((SECONDS + 180))
while true; do
  compgen -G "${H3_ATTEMPT_DIR}/failure.*" > /dev/null && exit 125
  (( stop_requested == 0 && SECONDS < deadline )) || exit 125
  ready=0
  for (( peer=0; peer<H3_NNODES; peer++ )); do
    [[ ! -f "${H3_ATTEMPT_DIR}/nodes/${peer}/ready" ]] || ready=$((ready + 1))
  done
  (( ready != H3_NNODES )) || break
  sleep 1
done
setsid env H3_ATTEMPT_NONCE="${H3_ATTEMPT_NONCE}" H3_ACCEPTANCE_ROOT="${H3_ACCEPTANCE_ROOT}" \
  H3_ATTEMPT_DIR="${H3_ATTEMPT_DIR}" H3_CHECKOUT="${H3_CHECKOUT}" H3_PYTHON="${H3_PYTHON}" \
  H3_NODE_RANK="${node_rank}" bash "${H3_CHILD_SCRIPT}" > "${node_dir}/child.log" 2>&1 9>&- &
child_pid=$!
flowart_gpu_lease_bind_process_group "${child_pid}"
owner_event "started child_pid=${child_pid}"
deadline=$((SECONDS + H3_MAX_SECONDS))
last_progress=0
while kill -0 "${child_pid}" 2>/dev/null; do
  if compgen -G "${H3_ATTEMPT_DIR}/failure.*" > /dev/null || (( stop_requested != 0 )); then
    owner_event 'peer failure or stop signal; draining this attempt'
    exit 125
  fi
  (( SECONDS < deadline )) || exit 124
  if (( SECONDS - last_progress >= 30 )); then
    owner_event "child alive pid=${child_pid}; inspect nodes/${node_rank}/child.log and per-rank logs"
    timeout -k 2 20 /usr/bin/python3 "${control}" \
      --lease "${FLOWART_GPU_LEASE_OWNER}" --owner-pid "$$" \
      --owner-start-ticks "${owner_start}" --generation "${FLOWART_GPU_LEASE_GENERATION}" \
      --nonce "${H3_ATTEMPT_NONCE}" --root "${H3_ACCEPTANCE_ROOT}" --observe \
      >> "${node_dir}/metrics.jsonl" 2>> "${node_dir}/metrics.stderr" || true
    tail -c 1500 "${node_dir}/child.log"
    printf '\n'
    last_progress=${SECONDS}
  fi
  sleep 1
done
status=0
wait "${child_pid}" || status=$?
owner_event "child finished exit=${status}"
exit "${status}"
