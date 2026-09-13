#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-14B}
DATA_PATH=${DATA_PATH:-open-r1/OpenR1-Math-220k/data}
training_max_length=${TRAINING_MAX_LENGTH:-32768}
bs=${BS:-32}
headpooling_type=${HEADPOOLING_TYPE:-"Qproj"}
gate_type=${GATE_TYPE:-"Kmaxminavg"}
gate_hidden_size=${SEERATTN_GATE_HIDDEN_SIZE:-128}
blocksize=${SEERATTN_BLOCK_SIZE:-64}
topk=${TOPK_SIZE:-31}
router_lr=${ROUTER_LR:-1e-3}
router_lr_decay_style=${ROUTER_LR_DECAY_STYLE:-cosine}
router_lr_warmup_ratio=${ROUTER_LR_WARMUP_RATIO:-0.03}
router_lr_min=${ROUTER_LR_MIN:-1e-5}
router_weight_decay=${ROUTER_WEIGHT_DECAY:-0.0}
use_qk_norm=${USE_QK_NORM:-true}
use_rope=${USE_ROPE:-true}

OUTPUT_DIR=${OUTPUT_DIR:-./outputs/simple_sparse_attention/Qwen3-14B-openr1-math-b${blocksize}k${topk}}
export LOG_PATH="${OUTPUT_DIR}/log.txt"

pixi run bash scripts/train/train.sh -m sas.train.trainer \
    configs/train_sparse.yaml \
    --model.model_path ${MODEL_PATH} \
    --model.tokenizer_path ${MODEL_PATH} \
    --data.train_path ${DATA_PATH} \
    --data.text_keys messages \
    --data.data_type conversation \
    --data.chat_template chatml \
    --data.max_seq_len ${training_max_length} \
    --data.num_workers 16 \
    --data.datasets_type mapping \
    --train.output_dir ${OUTPUT_DIR} \
    --train.global_batch_size ${bs} \
    --train.micro_batch_size 4 \
    --train.save_steps 1000 \
    --train.num_train_epochs 1 \
    --train.use_wandb true \
    --train.wandb_project sas \
    --train.wandb_name simple_sparse_attention-Qwen3-14B-openr1-math-b${blocksize}k${topk} \
    --sparse.sparse_mod simple_sparse_attention \
    --sparse.block_size ${blocksize} \
    --sparse.topk ${topk} \
    --sparse.gate_hidden_size ${gate_hidden_size} \
    --sparse.q_head_pooling_type ${headpooling_type} \
    --sparse.k_pooling_names ${gate_type} \
    --sparse.use_qk_norm ${use_qk_norm} \
    --sparse.use_rope_for_gate ${use_rope} \
    --sparse.router_lr ${router_lr} \
    --sparse.router_lr_decay_style ${router_lr_decay_style} \
    --sparse.router_lr_warmup_ratio ${router_lr_warmup_ratio} \
    --sparse.router_lr_min ${router_lr_min} \
    --sparse.router_weight_decay ${router_weight_decay}
