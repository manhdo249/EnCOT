export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=ECRTM
NUM_TOPICS=50
DATASET=Biomedical

WANDB_PROJECT=ECRTM_Biomedical50_0502
for weight_ot_topic_cluster in 1 0.1 0
do
    for weight_ot_doc_cluster in 1 0.1 0
    do
        for weight_loss_ECR in 10.0 20.0 40.0 
        do
            for seed in $(seq 0 2)
            do 
                CUDA_VISIBLE_DEVICES=0 python main.py  \
                                    --model $MODEL_NAME \
                                    --weight_ot_topic_cluster $weight_ot_topic_cluster \
                                    --weight_ot_doc_cluster $weight_ot_doc_cluster \
                                    --weight_loss_ECR $weight_loss_ECR \
                                    --pretrained_WE \
                                    --num_topics $NUM_TOPICS \
                                    --dataset $DATASET \
                                    --seed $seed \
                                    --wandb_on \
                                    --wandb_prj $WANDB_PROJECT \
                                    --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_weight_ot_topic_cluster${weight_ot_topic_cluster}_weight_ot_doc_cluster${weight_ot_doc_cluster}_weight_loss_ECR${weight_loss_ECR}_seed${seed}" \
                                    --verbose
            done
        done
    done
done 