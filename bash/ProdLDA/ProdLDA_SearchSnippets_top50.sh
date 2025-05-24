#!/bin/bash
# DONE

set -e 


MODEL_NAME=ProdLDA
NUM_TOPICS=50
DATASET=SearchSnippets

WANDB_PROJECT=ShortTextTM_240919

for dropout in 0.1 0.2 0.3
do
    for seed in $(seq 0 10)
    do
        CUDA_VISIBLE_DEVICES=0 python main.py  \
                            --model $MODEL_NAME \
                            --dropout $dropout \
                            --num_topics $NUM_TOPICS \
                            --dataset $DATASET \
                            --seed $seed \
                            --wandb_on \
                            --wandb_prj $WANDB_PROJECT \
                            --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_dropout${dropout}_seed${seed}" \
                            --verbose 
    done
done