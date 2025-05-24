#!/bin/bash
# DONE

set -e 

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi


MODEL_NAME=FASTopic
NUM_TOPICS=50
DATASET=Biomedical

WANDB_PROJECT=ShortTextTM_240919

for seed in $(seq 0 10)
do
    python main.py  \
        --model $MODEL_NAME \
        --num_topics $NUM_TOPICS \
        --dataset $DATASET \
        --seed $seed \
        --wandb_on \
        --wandb_prj $WANDB_PROJECT \
        --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_seed${seed}" \
        --verbose 
done