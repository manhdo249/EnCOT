import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

class ETM(nn.Module):
    '''
        Topic Modeling in Embedding Spaces. TACL 2020

        Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei.
    '''
    def __init__(self, vocab_size, embed_size=200, num_topics=50, en_units=800, dropout=0., pretrained_WE=None, train_WE=False, 
                 num_clusters=30, weight_ot_doc_cluster=1.0, weight_ot_topic_cluster=1.0):
        super().__init__()

        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = nn.Parameter(torch.randn((num_topics, self.word_embeddings.shape[1])))

        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.ReLU(),
            nn.Linear(en_units, en_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

        # OT loss initialization
        self.num_clusters = num_clusters
        self.weight_ot_doc_cluster = weight_ot_doc_cluster
        self.weight_ot_topic_cluster = weight_ot_topic_cluster

        self.cluster_embeddings = nn.Parameter(torch.randn(num_clusters, embed_size))

        self.ot_loss_fn_doc_cluster = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")
        self.ot_loss_fn_topic_cluster = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="tensorized")

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        e1 = self.encoder1(x)
        return self.fc21(e1), self.fc22(e1)

    def get_theta(self, x):
        # Warn: normalize the input if use Relu.
        # https://github.com/adjidieng/ETM/issues/3
        norm_x = x / x.sum(1, keepdim=True)
        mu, logvar = self.encode(norm_x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)
        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def get_beta(self):
        beta = F.softmax(torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1)
        return beta

    def compute_ot_loss_doc_cluster(self, doc_theta):
        doc_embeddings = torch.matmul(doc_theta, self.topic_embeddings)  # (batch_size, embed_size)
        ot_loss = self.ot_loss_fn_doc_cluster(doc_embeddings, self.cluster_embeddings)
        return ot_loss

    def compute_ot_loss_topic_cluster(self):
        ot_loss = self.ot_loss_fn_topic_cluster(self.topic_embeddings, self.cluster_embeddings)
        return ot_loss

    def forward(self, x, avg_loss=True):
        theta, mu, logvar = self.get_theta(x)
        beta = self.get_beta()
        recon_x = torch.matmul(theta, beta)

        recon_loss = self.loss_function(x, recon_x, mu, logvar, avg_loss)

        # Compute OT losses
        ot_loss_doc_cluster = self.compute_ot_loss_doc_cluster(theta)
        ot_loss_topic_cluster = self.compute_ot_loss_topic_cluster()

        total_loss = recon_loss + self.weight_ot_doc_cluster * ot_loss_doc_cluster + self.weight_ot_topic_cluster * ot_loss_topic_cluster

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'ot_loss_doc_cluster': ot_loss_doc_cluster,
            'ot_loss_topic_cluster': ot_loss_topic_cluster
        }

    def loss_function(self, x, recon_x, mu, logvar, avg_loss=True):
        recon_loss = -(x * (recon_x + 1e-12).log()).sum(1)
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        loss = (recon_loss + KLD)

        if avg_loss:
            loss = loss.mean()

        return loss
