import os
import torch
import numpy as np
import torch.nn as nn
import logging
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
import torch.nn.init as torch_init
import torch.nn.functional as F
import math
from tqdm import tqdm
import json
from src.transformerEncoder import make_model
from src.Attention import *


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def normalize(tensor):
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]

    range_vals = max_val - min_val
    range_vals[range_vals == 0] = 1  

    return (tensor - min_val) / range_vals

class TrainingModel(pl.LightningModule):
    """
    train definition
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss weight
        self.cfg = cfg
        self.min_loss = float('inf')
        self._logger = logging.getLogger("log_model")
        self.optimizer = None
        self.scheduler = None
        self.initial_optimizer_state_dict = None
        self.restart_epoch = 0

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log('step_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return loss.cpu().item()

    def validation_epoch_end(self, outputs) -> None:
        loss = np.mean([batch_loss for batch_loss in outputs])
        self.log('epoch_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        print('loss:', loss)
        if loss < self.min_loss:
            if self.current_epoch != 0: self.min_loss = loss
            torch.save(self.state_dict(),
                       os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))               
            print('model saved.')
        torch.save(self.state_dict(),
                    os.path.join(self.args.model_save_path, 'new.bin'))
        

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        self._logger.info('Test.')
        self.validation_epoch_end(outputs)

    def on_epoch_end(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True, prog_bar=True, logger=True)

    def optim_lr_lambda(self, epoch):
        if self.restart_epoch:
            return min((epoch + 1 - self.restart_epoch) ** -0.5, (epoch + 1 - self.restart_epoch) * self.args.warmup_epochs ** (-1.5))
        else:
            return min((epoch + 1) ** -0.5, (epoch + 1) * self.args.warmup_epochs ** (-1.5))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = LambdaLR(optimizer,
                             lr_lambda = self.optim_lr_lambda,
                             last_epoch=-1)

        self.initial_optimizer_state_dict = optimizer.state_dict()
        self.optimizer = optimizer
        self.scheduler = scheduler

        return [self.optimizer], [self.scheduler]




class Chimera(TrainingModel):
    def __init__(self, cfg, embedding_dict, input_size = 300, hidden_size = 128):
        super().__init__(cfg)
        self.args = cfg
        self.embedding_dict = embedding_dict
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = -1)
        self.apply(weight_init)
        self.hidden_size = hidden_size
        self.num_directions = 2
        self.attention_size = self.hidden_size

        self.rca_encoder = nn.GRU(input_size= input_size, hidden_size = hidden_size, num_layers = 2,
                          batch_first=True, bidirectional=True, dropout = 0.1)

        self.ad_encoder = nn.GRU(input_size= input_size, hidden_size = hidden_size, num_layers = 2,
                          batch_first=True, bidirectional=True, dropout = 0.1)    

        self.shared_encoder = nn.GRU(input_size= input_size, hidden_size = hidden_size, num_layers = 2,
                          batch_first=True, bidirectional=True, dropout = 0.1)    

        self.ce_loss = nn.CrossEntropyLoss()
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 2)

        self.atten_guide = Parameter(torch.Tensor(256))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim = 256, tensor_2_dim = 256)

        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.weights = [0.0] * 20
        for i in range(20):
            self.weights[i] = 1. / (i / 3 + 1)
        self.weights = torch.Tensor(self.weights).to(self.args.hard_device)
        self.weights = self.weights.expand(256, -1)

    def deal_batch(self, src, ad_label, rca_label):
        src_embedding = []
        for seq in src:
                seq_embedding = [self.embedding_dict[str(word)] for word in seq]
                src_embedding.append(seq_embedding)    

        src_embedding = np.array(src_embedding)
        src_embedding = torch.from_numpy(src_embedding).float()

        ad_label_tensor = []
        for batch in ad_label:
            if int(batch) == 0: ad_label_tensor.append(0)
            else: ad_label_tensor.append(1)

        ad_label_tensor = np.array(ad_label_tensor)
        ad_label_tensor = torch.from_numpy(ad_label_tensor).long()
    

        rca_label_tensor = []
        for seq in rca_label:
                seq_embedding = [int(word) for word in seq]
                rca_label_tensor.append(seq_embedding)    

        rca_label_tensor = np.array(rca_label_tensor)
        rca_label_tensor = torch.from_numpy(rca_label_tensor).float()  

    
        src_embedding = src_embedding.permute(1, 0, -1).to(self.args.hard_device)
        ad_label_tensor = ad_label_tensor.to(self.args.hard_device)
        rca_label_tensor = rca_label_tensor.transpose(0, 1).to(self.args.hard_device)

        
        return src_embedding, ad_label_tensor, rca_label_tensor

    def mlp(self, inputs):
        x = self.relu(self.fc1(inputs))
        if self.args.mode == 'train': x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if self.args.mode == 'train': x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        if self.args.mode == 'train': x = self.dropout(x)
        return x.view(x.shape[0], -1)

    def classifier(self, inputs):
        x = self.relu(self.fc4(inputs))
        if self.args.mode == 'train': x = self.dropout(x)
        x = self.relu(self.fc5(x))
        if self.args.mode == 'train': x = self.dropout(x)
        x = self.fc6(x)
        return x.view(x.shape[0], -1)

    def ranking_loss(self, n_bag_score, an_bag_score):
        loss = torch.tensor(0., requires_grad=True)
        bs = n_bag_score.shape[0]

        for i in range(bs):
            tmp = None
            maxn = torch.max(n_bag_score[i])

            an = torch.max(an_bag_score[i], dim = 0).values
            tmp = F.relu(1.-an+maxn)

            loss = loss + tmp

        return loss / bs

    def Diff_loss(self, input1, input2):
        feature_size = input1.size(-1)
        input1 = input1.reshape(-1, feature_size)
        input2 = input2.reshape(-1, feature_size)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

    def js_div(self, p_output, q_output, get_softmax=True):
            KLDivLoss = nn.KLDivLoss(reduction='batchmean')
            if get_softmax:
                p_output = F.softmax(p_output, dim = -1)
                q_output = F.softmax(q_output, dim = -1)
            log_mean_output = ((p_output + q_output) / 2).log()

            return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


    def attention_net(self, output, device = None):
        device = self.args.device
        output_reshape = torch.Tensor.reshape(output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))

        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, 20])
        
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])

        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, 20])
        
        return alphas_reshape

    def cal_weight(self, x):
        sorted_indices = torch.argsort(x, dim = -1, descending = True).to(self.args.hard_device)
        original_positions = torch.empty_like(sorted_indices).to(self.args.hard_device)
        rows = torch.arange(sorted_indices.size(0)).view(-1, 1).to(self.args.hard_device)  
        original_positions[rows, sorted_indices] = torch.arange(sorted_indices.size(-1)).to(self.args.hard_device)
        weighted_tensor = self.weights.gather(-1, original_positions).to(self.args.hard_device)
        #return self.softmax(weighted_tensor * x)
        return weighted_tensor * x

    def forward(self, batch):
        src, ad_label, rca_label = batch
        bag_src, ad_label, rca_label = self.deal_batch(src, ad_label, rca_label)
        bs = len(bag_src) // 2
                    
        n_bag_src = bag_src[: bs]
        an_bag_src = bag_src[bs: ]
        n_ad_label = ad_label[: bs]
        an_ad_label = ad_label[bs: ]
        n_rca_label = rca_label[: bs]
        an_rca_label = rca_label[bs: ]

        n_rca_out, states = self.rca_encoder(n_bag_src)
        an_rca_out, states = self.rca_encoder(an_bag_src)

        n_ad_out, states = self.ad_encoder(n_bag_src)
        an_ad_out, states = self.ad_encoder(an_bag_src)

        n_shared_out, states = self.shared_encoder(n_bag_src)
        an_shared_out, states = self.shared_encoder(an_bag_src)        

        diff_loss = self.Diff_loss(n_rca_out, n_shared_out) + self.Diff_loss(an_rca_out, an_shared_out) + self.Diff_loss(n_ad_out, n_shared_out) + self.Diff_loss(an_ad_out, an_shared_out)





        n_rca_out = (n_rca_out + n_shared_out) / 2
        an_rca_out = (an_rca_out + an_shared_out) / 2

        n_ad_out = (n_ad_out + n_shared_out) / 2
        an_ad_out = (an_ad_out + an_shared_out) / 2    



        n_score = self.mlp(n_rca_out)
        an_score = self.mlp(an_rca_out)

        rk_loss = self.ranking_loss(n_score, an_score)




        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, bs)
        atten_guide = atten_guide.transpose(1, 0)
        n_mask = self.atten(atten_guide, n_ad_out)
        an_mask = self.atten(atten_guide, an_ad_out)
       


        jsd_loss = self.js_div(n_mask, n_score) + self.js_div(an_mask, an_score)

        n_mask = self.cal_weight(n_mask)
        an_mask = self.cal_weight(an_mask)



        n_z = torch.sum(n_mask.unsqueeze(-1) * n_ad_out, dim = 1)  
        an_z = torch.sum(an_mask.unsqueeze(-1) * an_ad_out, dim = 1)

        n_classifier_out = self.classifier(n_z)
        an_classifier_out = self.classifier(an_z)



        n_one_hot_label = torch.zeros([n_ad_label.shape[0], 2]).to(self.args.hard_device)
        n_one_hot_label.scatter_(1, n_ad_label.unsqueeze(-1), 1)

        an_one_hot_label = torch.zeros([an_ad_label.shape[0], 2]).to(self.args.hard_device)
        an_one_hot_label.scatter_(1, an_ad_label.unsqueeze(-1), 1)

        classifier_loss = self.ce_loss(n_classifier_out, n_one_hot_label) + self.ce_loss(an_classifier_out, an_one_hot_label)


        loss = classifier_loss + 2 * rk_loss   + 0.001 * diff_loss + 0.5 * jsd_loss

        self.log('classifier_loss', classifier_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('rk_loss', 2 * rk_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('diff_loss', 0.001 * diff_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('jsd_loss', jsd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss, (n_classifier_out, n_score, n_ad_label, n_rca_label)








