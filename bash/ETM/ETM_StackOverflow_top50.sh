export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=ETM
NUM_TOPICS=50
DATASET=StackOverflow

WANDB_PROJECT=ETM_new_StackOverflow50_2501


for dropout in 0.0 0.1
do
    for en_units in 200 100 50
    do
        for lr in 0.002 0.001
        do
            for seed in $(seq 0 3)
            do
                python main.py  \
                    --model $MODEL_NAME \
                    --pretrained_WE \
                    --train_WE \
                    --en_units $en_units \
                    --dropout $dropout \
                    --num_topics $NUM_TOPICS \
                    --lr $lr \
                    --dataset $DATASET \
                    --seed $seed \
                    --wandb_on \
                    --wandb_prj $WANDB_PROJECT \
                    --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_en_units${en_units}_dropout${dropout}_lr${lr}_seed${seed}" \
                    --verbose
            done
        done
    done
done