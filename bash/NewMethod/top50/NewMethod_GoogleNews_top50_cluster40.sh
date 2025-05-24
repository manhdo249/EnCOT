export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=NewMethod
NUM_TOPICS=50
NUM_CLUSTERS=40
SINKHORN_MAX_ITER=100
DATASET=GoogleNewsCluster
GLOBAL_DIR=umap_globalcluster200
WANDB_PROJECT=NewMethod_GoogleNews_top50_cluster40_4_12

for weight_ot_topic_cluster in 0.01 0.1 0.5 1 10 0
do
    for weight_ot_doc_cluster in 0.01 0.1 0.5 1 10 0
    do
        for weight_loss_ECR in 30.0 60.0 90.0 10.0 20.0 120.0
        do
            for alpha_noise in 0.001 0.01 0.1
            do
                for alpha_augment in 0.01 0.05 0.1 1 0.5 0.0
                do
                    for seed in $(seq 0 3) 
                    do 
                        python main.py  \
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
                            --global_dir $GLOBAL_DIR \
                            --seed $seed \
                            --wandb_on \
                            --wandb_prj $WANDB_PROJECT \
                            --wandb_name "${MODEL_NAME}_${DATASET}_num_clusters_${NUM_CLUSTERS}_top${NUM_TOPICS}_${GLOBAL_DIR}_weight_ot_topic_cluster${weight_ot_topic_cluster}_weight_ot_doc_cluster${weight_ot_doc_cluster}_ECR${weight_loss_ECR}_alpha_noise${alpha_noise}_alpha_augment${alpha_augment}_seed${seed}" \
                            --verbose
                    done
                done
            done
        done
    done
done
