#!/bin/bash
# Running
set -e 

export WANDB_API_KEY=your_api_key_here  # Replace by your API key

MODEL_NAME=TSCTM
NUM_TOPICS=50
DATASET=GoogleNews

WANDB_PROJECT=ShortTextTM_240919

for weight_contrast in 1.0
do
    for temperature in 0.5
    do
        for seed in $(seq 0 2)
        do 
            echo 2 | CUDA_VISIBLE_DEVICES=0 python main.py  \
                --model $MODEL_NAME \
                --weight_contrast $weight_contrast \
                --temperature $temperature \
                --num_topics $NUM_TOPICS \
                --dataset $DATASET \
                --seed $seed \
                --wandb_on \
                --wandb_prj $WANDB_PROJECT \
                --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_weight_contrast${weight_contrast}_temperature${temperature}_seed${seed}" \
                --verbose
        done
    done
done
