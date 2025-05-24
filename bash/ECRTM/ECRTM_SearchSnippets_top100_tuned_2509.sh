#!/bin/bash
# ATUNG

set -e 


if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi


MODEL_NAME=ECRTM
NUM_TOPICS=100
DATASET=SearchSnippets

WANDB_PROJECT=ShortTextTM_240919


for weight_loss_ECR in 40.0 20.0 50.0
do
    for seed in $(seq 0 5)
    do 
        python main.py  \
            --model $MODEL_NAME \
            --weight_loss_ECR $weight_loss_ECR \
            --pretrained_WE \
            --num_topics $NUM_TOPICS \
            --dataset $DATASET \
            --seed $seed \
            --wandb_on \
            --wandb_prj $WANDB_PROJECT \
            --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_weight_loss_ECR${weight_loss_ECR}_seed${seed}" \
            --verbose
    done
done