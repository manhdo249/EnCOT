export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=NewMethod
NUM_TOPICS=100
NUM_CLUSTERS=30
SINKHORN_MAX_ITER=100
DATASET=SearchSnippets
WANDB_PROJECT=NewMethod_SearchSnippets_top100_cluster30_7_12

for weight_ot_topic_cluster in 0.01 1 0
do
    for weight_ot_doc_cluster in 0.01 1 
    do
        for weight_loss_ECR in 30.0 
        do
            for alpha_noise in 0.001 
            do
                for alpha_augment in 0.01 
                do
                    for seed in $(seq 0 3)
                    do 
                        CUDA_VISIBLE_DEVICES=0 python main.py  \
                                            --model $MODEL_NAME \
                                            --weight_loss_ECR $weight_loss_ECR \
                                            --weight_ot_topic_cluster $weight_ot_topic_cluster \
                                            --weight_ot_doc_cluster $weight_ot_doc_cluster \
                                            --sinkhorn_max_iter $SINKHORN_MAX_ITER \
                                            --alpha_noise $alpha_noise \
                                            --alpha_augment $alpha_augment \
                                            --pretrained_WE \
                                            --num_topics $NUM_TOPICS \
                                            --num_clusters $NUM_CLUSTERS \
                                            --dataset $DATASET \
                                            --seed $seed \
                                            --wandb_on \
                                            --wandb_prj $WANDB_PROJECT \
                                            --wandb_name "${MODEL_NAME}_${DATASET}_num_clusters_${NUM_CLUSTERS}_top${NUM_TOPICS}_weight_ot_topic_cluster${weight_ot_topic_cluster}_weight_ot_doc_cluster${weight_ot_doc_cluster}_weight_loss_ECR${weight_loss_ECR}_alpha_noise${alpha_noise}_alpha_augment${alpha_augment}_seed${seed}" \
                                            --verbose
                    done
                done
            done
        done
    done
done
