#!/bin/bash
# Done

set -e 

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

MODEL_NAME=KNNTM
NUM_TOPICS=50
DATASET=GoogleNews
WANDB_PROJECT=ShortTextTM_240919
alpha=1.0
num_k=20
eta=0.5
rho=0.5

for seed in $(seq 0 2)
do
    python main.py  \
        --model $MODEL_NAME \
        --alpha $alpha \
        --num_k $num_k \
        --eta $eta \
        --rho $rho \
        --num_topics $NUM_TOPICS \
        --dataset $DATASET \
        --p_epochs 20 \
        --seed $seed \
        --wandb_on \
        --wandb_prj $WANDB_PROJECT \
        --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_alpha${alpha}_num_k${num_k}_eta${eta}_rho${rho}_seed${seed}" \
        --verbose
done