#!/bin/bash




MODEL_VERSION=llava_loraft_dpo_our_ocrvqa8kfilter_diffu500_textvqa8kfilter_diffu500_r1024_a2048

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export MASTER_PORT=$PORT
echo "The Used MASTER_PORT: ${MASTER_PORT}"  




MODEL=/home/zhengzhixiao/model2/llava-v1.5-7b

CODE=catpo.py



TOKEN_REWARD_PATH=/data1/zhengzx/data/all5byrv-pe-5428yer2-N3-ld05ly19t22-th05mean-tok_TRUE.json 




LR=1.4e-6
NUM_EPOCH=6
TRAIN_BATCH_SIZE=1
GRAD_ACCUM=32
WEIGHT_DECAY=0.05   
WARMUP_RATIO=0.03
BETA=0.1

CLIP_NORM=1.0   
LAMBDA_KL=0.03  
LAMBDA_KL_WARMUP_RATIO=0.0      


#===GPU number===
GPU_TO_USE=3

#===save directory===
SAVE_DIR=tdp3-KL-PE-tok-7B-51118v1_r2_1_4e6ep6-wd005wr003-bt01-lmKL003_N3




COMMENT=see-the-left-comment


echo "使用的 GPU: ${GPU_TO_USE}"  
echo "使用的 计算代码: ${CODE}"  
echo "使用的 奖励值JSON: ${TOKEN_REWARD_PATH}"
echo "使用的 学习参数 LR/EPOCH/BATCH/ACC/WD/WR/BETA: ${LR} ${NUM_EPOCH} ${TRAIN_BATCH_SIZE} ${GRAD_ACCUM} ${WEIGHT_DECAY} ${WARMUP_RATIO} ${BETA}"
echo "使用的 其他参数 CLIP_NORM/LAMBDA_KL/LAMBDA_KL_WARMUP_RATIO: ${CLIP_NORM} ${LAMBDA_KL} ${LAMBDA_KL_WARMUP_RATIO}"
echo "使用的 模型: ${MODEL}"  
echo "使用的 保存路径: ${SAVE_DIR}" 
echo "备注: PE tok version  ${COMMENT}"


export WANDB_DISABLED=true
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1


deepspeed --include=localhost:${GPU_TO_USE} --master_port=${MASTER_PORT} ${CODE} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --deepspeed seva/scripts/zero3.json \
    --model_name_or_path ${MODEL} \
    --parquet_data_path /data1/zhengzx/dataset/RLHF-V-Dataset/RLHF-V-Dataset.parquet \
    --version v1 \
    --token_weights_path ${TOKEN_REWARD_PATH} \
    --vision_tower /home/zhengzhixiao/model/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data1/zhengzx/ckpts_val/${SAVE_DIR} \
    --learning_rate ${LR} \
    --num_train_epochs ${NUM_EPOCH} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${MODEL_VERSION} \
    --max_grad_norm ${CLIP_NORM} \
    --lambda_kl ${LAMBDA_KL} \
    --lambda_kl_warm_ratio ${LAMBDA_KL_WARMUP_RATIO} \
    --beta ${BETA}