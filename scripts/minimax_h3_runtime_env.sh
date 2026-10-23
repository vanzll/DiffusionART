set +u
source /m2v_intern2/zhenglin/my_bashrc.sh
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_PRELOAD CUDA_VISIBLE_DEVICES
unset NCCL_CUMEM_ENABLE NCCL_CUMEM_HOST_ENABLE NCCL_P2P_LEVEL NCCL_P2P_DIRECT_DISABLE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VIRTUAL_ENV="${H3_PYTHON%/bin/python}"
export PATH="${VIRTUAL_ENV}/bin:${H3_ACCEPTANCE_ROOT}/tools/ffmpeg-6.0-amd64-static:/usr/local/mpi/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export XDG_CACHE_HOME="${H3_ACCEPTANCE_ROOT}/cache" TMPDIR="${H3_ACCEPTANCE_ROOT}/tmp"
export HF_HOME="${H3_ACCEPTANCE_ROOT}/cache/huggingface"
export WANDB_DIR="${H3_ATTEMPT_DIR}/wandb" WANDB_CACHE_DIR="${H3_ACCEPTANCE_ROOT}/cache/wandb"
export WANDB_CONFIG_DIR="${H3_ACCEPTANCE_ROOT}/cache/wandb-config"
export WANDB_DATA_DIR="${H3_ACCEPTANCE_ROOT}/cache/wandb-data"
export WANDB_ARTIFACT_DIR="${H3_ACCEPTANCE_ROOT}/cache/wandb-artifacts"
export PET_LOG_DIR="${H3_ATTEMPT_DIR}/logs/torchrun/node-${H3_NODE_RANK}"
export PET_REDIRECTS=3 PET_TEE=3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NCCL_NET='NCCL RDMA Plugin v9'
export NCCL_NET_PLUGIN=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0 NCCL_NVLS_ENABLE=0 NCCL_SHM_DISABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=0 NCCL_IB_GID_INDEX=3 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=20 NCCL_NET_OVERHEAD=1000
export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,P2P
export NCCL_DEBUG_FILE="${H3_ATTEMPT_DIR}/logs/nccl.%h.%p.log"
export NCCL_IB_HCA
NCCL_IB_HCA="$(timeout -k 2 10 /usr/bin/python3 "${H3_CHECKOUT}/scripts/minimax_h3_process_control.py" --hcas)"
cd "${H3_CHECKOUT}"
