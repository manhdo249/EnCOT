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


for weight_loss_ECR in 30.0 40.0 20.0 50.0
do
    for seed in $(seq 0 5)
    do
        for lr in 0.002 0.001 0.0005
        do
            python main.py  \
                --model $MODEL_NAME \
                --weight_loss_ECR $weight_loss_ECR \
                --pretrained_WE \
                --num_topics $NUM_TOPICS \
                --dataset $DATASET \
                --lr $lr \
                --seed $seed \
                --wandb_on \
                --wandb_prj $WANDB_PROJECT \
                --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_en_units${en_units}_weight_loss_ECR${weight_loss_ECR}_lr${lr}_seed${seed}" \
                --verbose
        done
    done
done


for weight_loss_ECR in 30.0 40.0 20.0 50.0
do
    for seed in $(seq 0 5)
    do
        for en_units in 100 75 50 25
        do
            for lr in 0.002 0.001
            do
                python main.py  \
                    --model $MODEL_NAME \
                    --weight_loss_ECR $weight_loss_ECR \
                    --pretrained_WE \
                    --num_topics $NUM_TOPICS \
                    --dataset $DATASET \
                    --seed $seed \
                    --lr $lr\
                    --en_units $en_units\
                    --wandb_on \
                    --wandb_prj $WANDB_PROJECT \
                    --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_en_units${en_units}_weight_loss_ECR${weight_loss_ECR}_lr${lr}_seed${seed}" \
                    --verbose
            done
        done
    done
done