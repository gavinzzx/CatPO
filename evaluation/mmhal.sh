#!/usr/bin/env bash
################################################################################
# One‑click generation + GPT‑4 evaluation for MMHal‑Bench
################################################################################

PORT=$(python - <<'PY'
import socket, random
s = socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
export MASTER_PORT=$PORT
echo "The used MASTER_PORT: $MASTER_PORT"

# GPT‑4
API_KEY="xxx"


MODEL_VERSION=catpo-7b


# ======Whcih GPU======
GPU_TO_USE=2

# ====== (3)model-path======
MODEL_PATH=/home/gavin/ckpts_val/${MODEL_VERSION}

# ====== (4)base model======
MODEL_BASE=/home/gavin/model2/llava-v1.5-7b


#==================================
export CUDA_VISIBLE_DEVICES=$GPU_TO_USE
echo "The used GPU: $GPU_TO_USE"
echo "The used MODEL_VERSION: $MODEL_VERSION"
echo "The used MODEL_PATH: $MODEL_PATH"
echo "The used base model: $MODEL_BASE"  
OUT_DIR=/home/gavin/bmk/MMHal-Bench/output/${MODEL_VERSION}
mkdir -p ${OUT_DIR}
RESP_JSON=${OUT_DIR}/mmhal_${MODEL_VERSION}_6.json

# ------------------------ 1. Reasoning ------------------------------------------
torchrun --nproc_per_node 1 --master_port ${MASTER_PORT} evaluation/get_response.py \
  --input /home/gavin/bmk/MMHal-Bench/response_template.json \
  --output ${RESP_JSON} \
  --image-root /home/gavin/bmk/MMHal-Bench/images \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE}

# ------------------------ 2. GPT‑4 Evaluation ---------------------------------------
EVAL_JSON=${OUT_DIR}/mmhal_${MODEL_VERSION}_gpt4_eval_1_3.json

python evaluation/mmhal_eval_gpt4.py \
  --response ${RESP_JSON} \
  --evaluation ${EVAL_JSON} \
  --api-key ${API_KEY} \
  --gpt-model gpt-4-turbo-2024-04-09 

echo ">>> Finished! The outcome saved in ${OUT_DIR}"
