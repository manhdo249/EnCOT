from geomloss import SamplesLoss
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ECR import ECR
from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance


class NewMethod(nn.Module):
    def __init__(self, vocab_size, num_topics=50, en_units=200, 
                 dropout=0., pretrained_WE=None, embed_size=200, 
                 beta_temp=0.2, weight_loss_ECR=50.0, 
                 sinkhorn_alpha=20.0, 
                 sinkhorn_max_iter=1000,
                 OT_max_iter=5000,
                 stopThr=0.005,
                 alpha_noise=0.01, alpha_augment=0.05,
                 num_clusters=30,
                 weight_ot_doc_cluster=1.0,
                 weight_ot_topic_cluster=1.0,):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.alpha_noise = alpha_noise
        self.alpha_augment = alpha_augment

        self.OT_max_iter = OT_max_iter
        self.stopThr = stopThr
        self.sinkhorn_alpha = sinkhorn_alpha

        self.num_clusters = num_clusters
        self.weight_ot_doc_cluster = weight_ot_doc_cluster
        self.weight_ot_topic_cluster = weight_ot_topic_cluster

        self.cluster_embeddings = nn.Parameter(torch.randn(num_clusters, embed_size))

        self.ot_loss_fn_doc_cluster = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")
        self.ot_loss_fn_topic_cluster = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")

        # Prior
        ## global docs
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False
        ## local doc noise
        self.doc_noise_mu = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.doc_noise_var = nn.Parameter(torch.ones_like(self.var2)*self.alpha_noise, requires_grad=False)

        # global encoder
        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False

        # global decoder
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        # local noise encoder
        self.fc11_noise = nn.Linear(vocab_size, en_units)
        self.fc12_noise = nn.Linear(en_units, en_units)
        self.fc21_noise = nn.Linear(en_units, num_topics)
        self.fc22_noise = nn.Linear(en_units, num_topics)
        self.fc1_noise_dropout = nn.Dropout(dropout)

        self.noise_mean_bn = nn.BatchNorm1d(num_topics)
        self.noise_mean_bn.weight.requires_grad = False
        self.noise_logvar_bn = nn.BatchNorm1d(num_topics)
        self.noise_logvar_bn.weight.requires_grad = False

        # word embedding
        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def global_encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.global_reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_global_KL(mu, logvar)

        return theta, loss_KL

    def global_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
    
    def compute_loss_global_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def noise_local_encode(self, input):
        e1 = F.softplus(self.fc11_noise(input))
        e1 = F.softplus(self.fc12_noise(e1))
        e1 = self.fc1_noise_dropout(e1)
        mu = self.noise_mean_bn(self.fc21_noise(e1))
        logvar = self.noise_logvar_bn(self.fc22_noise(e1))
        z = self.noise_local_reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_local_KL(mu, logvar)

        return theta, loss_KL

    def noise_local_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu 

    def compute_loss_local_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.doc_noise_var
        diff = mu - self.doc_noise_mu
        diff_term = diff * diff / self.doc_noise_var
        logvar_division = self.doc_noise_var.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division +  diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD
    
    def get_beta(self):
        dist = pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def get_theta(self, input):
        local_x = input[:, :self.vocab_size]
        global_x = input[:, self.vocab_size:]
        local_noise_theta, local_noise_loss_KL = self.noise_local_encode(local_x)
        global_theta, _ = self.global_encode(global_x)
        
        if self.training:
            return global_theta * local_noise_theta, _ , local_noise_loss_KL
        else:
            return global_theta * local_noise_theta


    def get_loss_ECR(self):
        cost = pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    def compute_ot_loss_doc_cluster(self, doc_theta):
        doc_embeddings = torch.matmul(doc_theta, self.topic_embeddings)  # (batch_size, embed_size)
        ot_loss = self.ot_loss_fn_doc_cluster(doc_embeddings, self.cluster_embeddings)
        return ot_loss

    def compute_ot_loss_topic_cluster(self):
        ot_loss = self.ot_loss_fn_topic_cluster(self.topic_embeddings, self.cluster_embeddings)
        return ot_loss


    def forward(self, input, is_ECR=True):
        local_x = input[:, :self.vocab_size]

        global_x = input[:, self.vocab_size:]
        local_noise_theta, local_noise_loss_KL = self.noise_local_encode(local_x)
        global_theta, _ = self.global_encode(global_x)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(global_theta * local_noise_theta, beta)), dim=-1)
        recon_loss = -((local_x + self.alpha_augment*global_x) * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + local_noise_loss_KL

        if is_ECR:
            loss_ECR = self.get_loss_ECR()
        else: 
            loss_ECR = 0

        ot_loss_doc_cluster = self.compute_ot_loss_doc_cluster(global_theta * local_noise_theta)
        ot_loss_topic_cluster = self.compute_ot_loss_topic_cluster()
            
        loss = loss_TM + loss_ECR + self.weight_ot_doc_cluster * ot_loss_doc_cluster +  self.weight_ot_topic_cluster * ot_loss_topic_cluster

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'ot_loss_doc_cluster': ot_loss_doc_cluster,
            'ot_loss_topic_cluster': ot_loss_topic_cluster,
        }

        return rst_dict
