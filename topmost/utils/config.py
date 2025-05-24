import argparse

def new_parser():
    return argparse.ArgumentParser(description="CLI for different shortTM settings")

# Loggings
def add_logging_argument(parser):
    parser.add_argument("--wandb_on", action="store_true", help="Log on wandb")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument('--wandb_prj', type=str, default='ShortTextTM')
    parser.add_argument("--verbose", action="store_true")


# Datasets
def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name, currently support datasets are: \
                            GoogleNews, SearchSnippets, \
                            StackOverflow, Biomedical")
    parser.add_argument('--global_dir', type=str, default="global")


# Models
def add_ProdLDA_arguments(subparser):
    subparser.add_argument('--dropout', type=float, default=0.0)
    subparser.add_argument('--en_units', type=int, default=200)

def add_TSCTM_arguments(subparser):
    subparser.add_argument('--temperature', type=float, default=0.5)
    subparser.add_argument('--weight_contrast', type=float, default=1.0)

def add_ECRTM_arguments(subparser):
    subparser.add_argument('--dropout', type=float, default=0.0)
    subparser.add_argument("--beta_temp", type=float, default=0.2)
    subparser.add_argument("--weight_loss_ECR", type=float, default=100.0)
    subparser.add_argument("--weight_ot_doc_cluster", type=float, default=1.0)
    subparser.add_argument("--weight_ot_topic_cluster", type=float, default=1.0)
    subparser.add_argument("--sinkhorn_alpha", type=float, default=20.0)
    subparser.add_argument("--sinkhorn_max_iter", type=int, default=1000)
    subparser.add_argument("--pretrained_WE", action="store_true")
    subparser.add_argument('--en_units', type=int, default=200)

def add_NewMethod_arguments(subparser):
    subparser.add_argument('--dropout', type=float, default=0.0)
    subparser.add_argument("--beta_temp", type=float, default=0.2)
    subparser.add_argument("--weight_loss_ECR", type=float, default=60.0)
    subparser.add_argument("--weight_ot_doc_cluster", type=float, default=1.0)
    subparser.add_argument("--weight_ot_topic_cluster", type=float, default=1.0)
    subparser.add_argument("--sinkhorn_alpha", type=float, default=20.0)
    subparser.add_argument("--sinkhorn_max_iter", type=int, default=100)
    subparser.add_argument("--alpha_noise", type=float, default=0.001)
    subparser.add_argument("--alpha_augment", type=float, default=0.05)
    subparser.add_argument("--pretrained_WE", action="store_true")

def add_KNNTM_arguments(subparser):
    subparser.add_argument('--alpha', type=float, default=1.0)
    subparser.add_argument('--num_k', type=int, default=20)
    subparser.add_argument('--eta', type=float, default=0.5)
    subparser.add_argument('--rho', type=float, default=0.5)

    subparser.add_argument("--p_epochs", type=int, default=20, help="Training epochs without augmentation")

def add_FASTopic_arguments(subparser):
    subparser.add_argument('--DT_alpha', type=float, default=3.0)
    subparser.add_argument('--TW_alpha', type=int, default=2.0)
    subparser.add_argument('--theta_temp', type=float, default=1.0)

def add_ETM_arguments(subparser):
    subparser.add_argument("--weight_ot_doc_cluster", type=float, default=1.0)
    subparser.add_argument("--weight_ot_topic_cluster", type=float, default=1.0)
    subparser.add_argument('--en_units', type=int, default=200)
    subparser.add_argument('--dropout', type=float, default=0.0)
    subparser.add_argument("--pretrained_WE", action="store_true")
    subparser.add_argument("--train_WE", action="store_true")


def add_model_argument(parser):

    parser.add_argument('--model', type=str, choices=['NewMethod', 'ProdLDA', 'TSCTM', 'ECRTM', 'KNNTM', 'FASTopic', 'ETM'], required=True,
                                help='Model name, current supports: NewMethod, ProdLDA, TSCTM, ECRTM, KNNTM, FASTopic, ETM')

    args, _ = parser.parse_known_args()



    model = args.model

    parser = argparse.ArgumentParser(
        description="CLI for different topic modeling approaches."
    )
    parser.add_argument('--model', type=str, choices=['NewMethod', 'ProdLDA', 'TSCTM', 'ECRTM', 'KNNTM', 'FASTopic', 'ETM'], required=True,
                        help='Model name, current supports: NewMethod, ProdLDA, TSCTM, ECRTM, KNNTM, FASTopic, ETM')

    if model == "ProdLDA":
        add_ProdLDA_arguments(parser)
    elif model == "TSCTM":
        add_TSCTM_arguments(parser)
    elif model == "ECRTM":
        add_ECRTM_arguments(parser)
    elif model == "NewMethod":
        add_NewMethod_arguments(parser)
    elif model == "KNNTM":
        add_KNNTM_arguments(parser)
    elif model == "FASTopic":
        add_FASTopic_arguments(parser)
    elif model == "ETM":
        add_ETM_arguments(parser)

    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_clusters', type=int, default=50)

    return parser


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train the model')
    parser.add_argument('--device', type=str, default='cuda',help='device to run the model, cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,help='learning rate scheduler, dont use if not needed, currently support: step')
    parser.add_argument('--lr_step_size', type=int, default=125, help='step size for learning rate scheduler')

def add_evaluation_argument(parser):
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--purity_threshold_for_cv', type=float, default=0.0)


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
