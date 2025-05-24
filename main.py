import os
import wandb
import topmost
from topmost.utils import seed, config, misc, log

RESULT_DIR = 'run'
DATA_DIR = 'data'

if __name__ == "__main__":

    ########################### Settings ####################################
    parser = config.new_parser()
    parser = config.add_model_argument(parser)
    config.add_dataset_argument(parser)
    config.add_training_argument(parser)
    config.add_evaluation_argument(parser)
    config.add_logging_argument(parser)
    args = parser.parse_args()

    seed.seedEverything(args.seed)

    current_time = misc.get_current_datetime()
    if not args.wandb_name:
        print(f"There is no wandb run name so current running name is {current_time}.")
        args.wandb_name = current_time

    current_run_dir = os.path.join(RESULT_DIR, args.wandb_prj, args.wandb_name)
    misc.create_folder_if_not_exist(current_run_dir)
    config.save_config(args, os.path.join(current_run_dir, "config.txt"))
    print(".. STARTING ..")
    print(args)

    logger = log.setup_logger('main', os.path.join(current_run_dir, "main.log"))
    wandb.init(name=args.wandb_name, project=args.wandb_prj, config=args, mode="online" if args.wandb_on else "offline")
    wandb.log({'time_stamp': current_time})

    ########################### Dataset ####################################
    if args.model == "NewMethod":
        dataset = topmost.data.BasicDatasetWithGlobal(dataset_dir=os.path.join(DATA_DIR, args.dataset),
                                                      global_dir=args.global_dir,
                                                      batch_size=args.batch_size,
                                                      read_labels=True,
                                                      device=args.device)
    elif args.model == "KNNTM":
        dataset = topmost.data.BasicDatasetWithIndex(dataset_dir=os.path.join(DATA_DIR, args.dataset),
                                                      batch_size=args.batch_size,
                                                      read_labels=True,
                                                      device=args.device)
    else:
        dataset = topmost.data.BasicDataset(dataset_dir=os.path.join(DATA_DIR, args.dataset), 
                                            batch_size=args.batch_size,
                                            read_labels=True, 
                                            device=args.device)
    

    ########################### Model and Training ####################################
    if args.model == "ProdLDA":
        model = topmost.models.ProdLDA(dataset.vocab_size, num_topics=args.num_topics, 
                                       dropout=args.dropout, en_units=args.en_units)
    elif args.model == "TSCTM":
        model = topmost.models.TSCTM(dataset.vocab_size, num_topics=args.num_topics,
                                     temperature=args.temperature, weight_contrast=args.weight_contrast)
    elif args.model == "ECRTM":
        model = topmost.models.ECRTM(dataset.vocab_size, num_topics=args.num_topics, dropout=args.dropout,
                                     beta_temp=args.beta_temp, weight_loss_ECR=args.weight_loss_ECR,
                                     sinkhorn_alpha=args.sinkhorn_alpha, sinkhorn_max_iter=args.sinkhorn_max_iter,
                                     pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None,
                                     en_units=args.en_units, num_clusters=args.num_clusters, 
                                     weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                                     weight_ot_topic_cluster=args.weight_ot_topic_cluster,)
    elif args.model == "NewMethod":
        model = topmost.models.NewMethod(dataset.vocab_size, num_topics=args.num_topics,
                                     num_clusters=args.num_clusters , dropout=args.dropout,
                                     beta_temp=args.beta_temp, weight_loss_ECR=args.weight_loss_ECR,
                                     weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                                     weight_ot_topic_cluster=args.weight_ot_topic_cluster,
                                     sinkhorn_alpha=args.sinkhorn_alpha, sinkhorn_max_iter=args.sinkhorn_max_iter,
                                     alpha_noise=args.alpha_noise, alpha_augment=args.alpha_augment,
                                     pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None)
    elif args.model == "KNNTM":
        model = topmost.models.KNNTM(dataset.vocab_size, len(dataset.train_data), dataset.train_data,
                                     os.path.join(DATA_DIR, args.dataset, "KNNTM", "M_cos.npz"), 
                                     os.path.join(DATA_DIR, args.dataset, "KNNTM", "M_coo.npz"), 
                                     alpha=args.alpha, num_k=args.num_k, eta=args.eta, rho=args.rho,
                                     num_topics=args.num_topics)
    elif args.model == "ETM":
        model = topmost.models.ETM(dataset.vocab_size, num_topics=args.num_topics, 
                                   en_units=args.en_units, dropout=args.dropout,
                                   weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                                   weight_ot_topic_cluster=args.weight_ot_topic_cluster, 
                                   pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None,
                                   train_WE=args.train_WE)

    elif args.model == "FASTopic":
        model = None
        
    if model is not None:
        model = model.to(args.device)

    # trainer
    if args.model == "KNNTM":
        trainer = topmost.trainers.KNNTMTrainer(model=model,
                                                dataset=dataset, 
                                                epochs=args.epochs,
                                                p_epoches=args.p_epochs,
                                                learning_rate=args.lr,
                                                batch_size=args.batch_size,
                                                num_top_words=args.num_top_word,
                                                lr_scheduler=args.lr_scheduler,
                                                lr_step_size=args.lr_step_size,
                                                verbose=args.verbose) 
    elif args.model == "FASTopic":
        trainer = topmost.trainers.FASTopicTrainer(dataset=dataset,
                                                   epochs=args.epochs,
                                                   num_topics=args.num_topics,
                                                   learning_rate=args.lr,
                                                   num_top_words=args.num_top_word,
                                                   preprocessing=None,
                                                   DT_alpha=args.DT_alpha,
                                                   TW_alpha=args.TW_alpha,
                                                   theta_temp=args.theta_temp,
                                                   verbose=args.verbose)
    else:
        trainer = topmost.trainers.BasicTrainer(model=model,
                                                dataset=dataset, 
                                                epochs=args.epochs,
                                                learning_rate=args.lr,
                                                batch_size=args.batch_size,
                                                num_top_words=args.num_top_word,
                                                lr_scheduler=args.lr_scheduler,
                                                lr_step_size=args.lr_step_size,
                                                verbose=args.verbose)
    
    trainer.train()

    ########################### Save ########################################
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(current_run_dir)
    top_words, top_words_path = trainer.save_top_words(current_run_dir)
    ########################### Evaluate ####################################
    # TD
    TD = topmost.evaluations.compute_topic_diversity(top_words)
    logger.info(f"TD: {TD:.5f}")
    wandb.log({"TD": TD})

    # Purity, NMI
    result = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
    logger.info(f"Purity: {result['Purity']:.5f}")
    wandb.log({"Purity": result['Purity']})

    logger.info(f"NMI: {result['NMI']:.5f}")
    wandb.log({"NMI": result['NMI']})

    # TC
    if result['Purity'] >= args.purity_threshold_for_cv:
        TCs, TC = topmost.evaluations.compute_topic_coherence_on_wikipedia(top_words_path)
        logger.info(f"TCs: {TCs}")
        logger.info(f"TC: {TC:.5f}")
        wandb.log({"TC": TC})

    print(".. FINISH ..")
    wandb.finish()
    print(args)