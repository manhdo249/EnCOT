#!/bin/bash
# DONE

set -e 

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

MODEL_NAME=ProdLDA
NUM_TOPICS=100
DATASET=Biomedical

WANDB_PROJECT=ShortTextTM_240919


for dropout in 0.3
do
    for seed in $(seq 6 20)
    do
        for en_units in 75
        do
            for lr in 0.002 0.001
            do
                python main.py  \
                    --model $MODEL_NAME \
                    --dropout $dropout \
                    --num_topics $NUM_TOPICS \
                    --dataset $DATASET \
                    --seed $seed \
                    --en_units $en_units \
                    --lr $lr \
                    --wandb_on \
                    --wandb_prj $WANDB_PROJECT \
                    --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_en_units${en_units}_dropout${dropout}_lr${lr}_seed${seed}" \
                    --verbose 
            done
        done
    done
done