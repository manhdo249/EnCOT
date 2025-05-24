export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=ECRTM
NUM_TOPICS=100
DATASET=Biomedical

WANDB_PROJECT=ECRTM_Biomedical100_0702

for weight_ot_topic_cluster in 1 0
do
    for weight_ot_doc_cluster in 1 
    do
        for weight_loss_ECR in 19.1 19.2 19.3 19.4 19.5 19.6 19.7 19.8 19.9 20.1 20.2 20.3 20.4 20.5 20.6 20.7 20.8 20.9
        do
            for seed in 0
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