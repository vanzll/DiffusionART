source /m2v_intern2/zhenglin/my_bashrc.sh
set -euo pipefail
source "${1:?missing attempt environment}"
mkdir -p "${H3_ATTEMPT_DIR}/logs"
printf '%s MPI start %s\n' "$(date -u +%FT%TZ)" "${H3_ATTEMPT_NONCE}"
status=0
/usr/local/mpi/bin/mpirun --allow-run-as-root --hostfile "${H3_HOSTFILE:-/etc/mpi/hostfile}" \
  --mca plm_rsh_args "-o StrictHostKeyChecking=yes -o UserKnownHostsFile=${H3_SSH_KNOWN_HOSTS}" \
  --mca orte_abort_on_non_zero_status 0 \
  -np "${H3_NNODES}" --map-by ppr:1:node --bind-to none \
  bash "${H3_CHECKOUT}/scripts/minimax_h3_node_owner.sh" "${H3_ATTEMPT_DIR}/launch.env" || status=$?
for (( node_rank=0; node_rank<H3_NNODES; node_rank++ )); do
  receipt="${H3_ATTEMPT_DIR}/nodes/${node_rank}/exitcode"
  if [[ ! -f "${receipt}" ]] || [[ "$(cat "${receipt}")" != 0 ]]; then
    status=1
  fi
done
if compgen -G "${H3_ATTEMPT_DIR}/failure.*" > /dev/null; then
  status=1
fi
printf '%s\n' "${status}" > "${H3_ATTEMPT_DIR}/mpi.exitcode"
printf '%s MPI exit %s\n' "$(date -u +%FT%TZ)" "${status}"
exit "${status}"
