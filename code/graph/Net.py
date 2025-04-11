import os
import time
import dgl
import numpy as np
import pandas as pd
from dgl.data import Subset
import torch
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from MOL_features import *


def reset_parameters(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            print(f"Initialized Linear layer weights with xavier_normal_")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                print(f"Initialized Linear layer bias with zeros_")

        # elif isinstance(module, nn.LSTM):
        #     for name, param in module.named_parameters():
        #         if 'weight_ih' in name:
        #             nn.init.xavier_normal_(param.data)
        #             print(f"Initialized LSTM weight_ih with xavier_normal_")
        #         elif 'weight_hh' in name:
        #             nn.init.orthogonal_(param.data)
        #             print(f"Initialized LSTM weight_hh with orthogonal_")
        #         elif 'bias' in name:
        #             nn.init.zeros_(param.data)
        #             print(f"Initialized LSTM bias with zeros_")
        #
        # elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        #     nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        #     print(f"Initialized Conv layer weights with kaiming_normal_")
        #     if module.bias is not None:
        #         nn.init.zeros_(module.bias)
        #         print(f"Initialized Conv layer bias with zeros_")

        elif isinstance(module, GraphConv):
            nn.init.xavier_normal_(module.weight)
            print(f"Initialized GraphConv layer weights with xavier_normal_")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                print(f"Initialized GraphConv layer bias with zeros_")


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = hidden_size ** 0.5

    def forward(self, query, key, value):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights


class SelfAttention(nn.Module):  # selfAttention操作对每个时间步的Query、Key、Value进行计算，并得到每个时间步的加权输出
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.key_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.value_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = ScaledDotProductAttention(hidden_size)

    def forward(self, lstm_output):
        query = self.query_layer(lstm_output)  # (batch, seq, hidden)
        key = self.key_layer(lstm_output)
        value = self.value_layer(lstm_output)
        attention_output, attention_weights = self.attention(query, key, value)
        return attention_output, attention_weights


# class BiLSTM_selfattn(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(BiLSTM_selfattn, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
#         self.attention = SelfAttention(hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)  # Output size matches hidden_size after attention
#
#     def forward(self, x):
#         x = x.unsqueeze(0)
#         h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
#         c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
#
#         lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, seq_length, hidden_size*2)
#         attn_out, attn_weights = self.attention(
#             lstm_out)  # attn_out: (batch_size, seq_length, hidden_size)   weights:(batch_size, seq_length, seq_length)
#
#         # Compute the mean over all time steps
#         attn_out = attn_out.mean(dim=1)  # (batch_size, hidden_size)
#
#         out = self.fc(attn_out)  # (batch_size, output_size)
#         return out, attn_weights


class BiLSTM_selfattn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM_selfattn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # 注意：输出大小与Attention后的hidden_size一致

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, seq_length, hidden_size*2)
        attn_out, attn_weights = self.attention(
            lstm_out)  # attn_out: (batch_size, seq_length, hidden_size)   weights:(batch_size, seq_length, seq_length)

        # # 取第一步和最后一步的输出，然后计算它们的平均值
        # first_step = attn_out[:, 0, :]  # 取第一步的输出 (batch_size, hidden_size)
        # last_step = attn_out[:, -1, :]  # 取最后一步的输出 (batch_size, hidden_size)
        # avg_out = (first_step + last_step) / 2  # 计算平均值 (batch_size, hidden_size)

        last_step = attn_out[:, -1, :]

        out = self.fc(last_step)  # (batch_size, output_size)
        return out, attn_weights


class GCN_BiLSTM_selfattn(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_BiLSTM_selfattn, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')
        if len(smiles_kmer.size()) != 2:
            smiles_kmer = smiles_kmer.unsqueeze(0)
        # print(smiles_GCNseq.shape)
        features = torch.cat((hg_seq, g_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, att_weights


class GCN_BiLSTM(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_BiLSTM, self).__init__()
        # Define a simple BiLSTM
        self.bilstm = nn.LSTM(input_size=17, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        # Apply BiLSTM to g_mol features
        g_mol, _ = self.bilstm(g_mol)  # (batch_size, seq_length, 256) as BiLSTM is bidirectional
        g_mol = g_mol.mean(dim=0, keepdim=True)  # 结果的形状为 (1, 256)

        # Graph convolution layers on g_seq
        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if len(smiles_kmer.size()) != 2:
            smiles_kmer = smiles_kmer.unsqueeze(0)

        # Concatenate features from graph, g_mol, and smiles_kmer
        features = torch.cat((hg_seq, g_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)

        # Pass through fully connected layers
        out = self.fc1(features)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out


class GCN_BiLSTM_selfattn_GCN1(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_BiLSTM_selfattn_GCN1, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)

        self.in_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if len(smiles_kmer.size()) != 2:
            smiles_kmer = smiles_kmer.unsqueeze(0)

        features = torch.cat((hg_seq, g_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)

        out = self.fc1(features)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out, att_weights


class GCN_BiLSTM_selfattn_GCN3(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_BiLSTM_selfattn_GCN3, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)

        self.in_feats_seq = 128
        self.h_feats_seq = 128
        self.o_feats = 256

        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv3_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv3_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)

        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if len(smiles_kmer.size()) != 2:
            smiles_kmer = smiles_kmer.unsqueeze(0)

        features = torch.cat((hg_seq, g_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)

        out = self.fc1(features)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out, att_weights


class EMMPTNet_unbalance(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(EMMPTNet_unbalance, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles, kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if smiles.dim() == 1:
            smiles = smiles.unsqueeze(0)
        if kmer.dim() == 1:
            kmer = kmer.unsqueeze(0)  # 扩展维度至[batch_size, 1]
        features = torch.cat((hg_seq, g_mol, smiles, kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, att_weights


class Without_GCN(nn.Module):
    def __init__(self):
        super(Without_GCN, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        # h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        # h_seq = F.relu(h_seq)
        # h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        # g_seq.ndata['h'] = h_seq
        # hg_seq = dgl.mean_nodes(g_seq, 'h')

        # print(smiles_GCNseq)
        # print(smiles_GCNseq.shape)
        smiles_kmer = smiles_kmer.unsqueeze(0)
        # print(smiles_GCNseq.shape)
        features = torch.cat((g_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, att_weights


class Without_MOL(nn.Module):
    def __init__(self):
        super(Without_MOL, self).__init__()
        # self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        # g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        smiles_kmer = smiles_kmer.unsqueeze(0)

        features = torch.cat((hg_seq, smiles_kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_heads, num_layers=1, dropout=0.1):
        super(BiLSTMWithAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # BiLSTM层
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)

        # 将BiLSTM的输出投影到embed_dim
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(0)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        # BiLSTM
        lstm_out, _ = self.bilstm(x, (h0, c0))  # (batch_size, seq_len, hidden_dim * 2)

        # 投影到embed_dim
        proj_out = self.fc(lstm_out)  # (batch_size, seq_len, embed_dim)

        # 注意力机制
        attn_output, attn_weights = self.multihead_attn(proj_out, proj_out, proj_out)  # 自注意力机制

        # 输出层
        attn_out = attn_output[:, -1, :]

        return attn_out, attn_weights


class GCN_BiLSTM_nnatt(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_BiLSTM_nnatt, self).__init__()
        self.bilstm_selfatt = BiLSTMWithAttention(input_dim=14, hidden_dim=128, embed_dim=256, num_heads=4,
                                                  num_layers=2, dropout=0.1)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)
        self.fc1 = nn.Linear(self.o_feats + 256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)

        h_seq = self.conv1_seq(g_seq, feat_seq)
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq)
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        # print(smiles_GCNseq)
        # print(smiles_GCNseq.shape)
        smiles_kmer = smiles_kmer.unsqueeze(0)
        # print(smiles_GCNseq.shape)
        features = torch.cat((hg_seq, g_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out, att_weights


class GCN_MSSK(nn.Module):  # MOL, STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_MSSK, self).__init__()
        self.in_feats_seq = 128  # 输入维度
        self.in_feats_mol = 17
        self.h_feats_seq = 128
        self.h_feats_mol = 128
        self.o_feats = 256
        self.conv1_mol = GraphConv(self.in_feats_mol, self.h_feats_mol, norm='none', allow_zero_in_degree=True)
        self.conv2_mol = GraphConv(self.h_feats_mol, self.o_feats, norm='none', allow_zero_in_degree=True)
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)
        self.fc1 = nn.Linear(self.o_feats * 2 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.bilstm_selfatt = BiLSTM_selfattn(input_size=14, hidden_size=128, num_layers=2, output_size=256)

    def forward(self, g_mol, g_seq, feat_mol, feat_seq, smiles_kmer):
        # m_mol, att_weights = self.bilstm_selfatt(m_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = dgl.add_self_loop(g_mol)
        h_mol = self.conv1_mol(g_mol, feat_mol, edge_weight=g_mol.edata['weight'])
        h_mol = F.relu(h_mol)
        h_mol = self.conv2_mol(g_mol, h_mol, edge_weight=g_mol.edata['weight'])
        g_mol.ndata['h'] = h_mol
        hg_mol = dgl.mean_nodes(g_mol, 'h')
        # g_seq = dgl.add_self_loop(g_seq)
        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        # print(smiles_GCNseq)
        # print(smiles_GCNseq.shape)
        smiles_kmer = smiles_kmer.unsqueeze(0)
        # print(smiles_GCNseq.shape)
        features = torch.cat((hg_seq, hg_mol, smiles_kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out


class GCN_BiLSTM_selfattn_withoutDes(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(GCN_BiLSTM_selfattn_withoutDes, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, g_mol, g_seq, feat_seq, smiles_kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        # print(smiles_GCNseq)
        # print(smiles_GCNseq.shape)
        # smiles_kmer = smiles_kmer.unsqueeze(0)
        # print(smiles_GCNseq.shape)
        features = torch.cat((hg_seq, g_mol), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out, att_weights


class EMMPTNet_unbalance_withoutGCN(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(EMMPTNet_unbalance_withoutGCN, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(256 + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles, kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)

        if smiles.dim() == 1:
            smiles = smiles.unsqueeze(0)
        if kmer.dim() == 1:
            kmer = kmer.unsqueeze(0)  # 扩展维度至[batch_size, 1]
        features = torch.cat((g_mol, smiles, kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, att_weights


class EMMPTNet_unbalance_withoutMOL(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(EMMPTNet_unbalance_withoutMOL, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 574 + 64, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles, kmer):
        # g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if smiles.dim() == 1:
            smiles = smiles.unsqueeze(0)
        if kmer.dim() == 1:
            kmer = kmer.unsqueeze(0)  # 扩展维度至[batch_size, 1]
        features = torch.cat((hg_seq, smiles, kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, hg_seq


class EMMPTNet_unbalance_withoutKmer(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(EMMPTNet_unbalance_withoutKmer, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 574, 575)
        self.fc2 = nn.Linear(575, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g_mol, g_seq, feat_seq, smiles, kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if smiles.dim() == 1:
            smiles = smiles.unsqueeze(0)
        if kmer.dim() == 1:
            kmer = kmer.unsqueeze(0)  # 扩展维度至[batch_size, 1]
        features = torch.cat((hg_seq, g_mol, smiles), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, att_weights


class EMMPTNet_unbalanced_withoutDes(nn.Module):  # MOL(bilstm), STRUCTURE, SEQ, KMER
    def __init__(self):
        super(EMMPTNet_unbalanced_withoutDes, self).__init__()
        self.bilstm_selfatt = BiLSTM_selfattn(input_size=17, hidden_size=128, num_layers=2, output_size=256)
        self.in_feats_seq = 128  # 输入维度
        self.h_feats_seq = 128
        self.o_feats = 256
        self.conv1_seq = GraphConv(self.in_feats_seq, self.h_feats_seq, norm='none', allow_zero_in_degree=True)
        self.conv2_seq = GraphConv(self.h_feats_seq, self.o_feats, norm='none', allow_zero_in_degree=True)

        self.fc1 = nn.Linear(self.o_feats + 256 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, g_mol, g_seq, feat_seq, smiles, kmer):
        g_mol, att_weights = self.bilstm_selfatt(g_mol)  # (1,256), (batch_size,  seq_length)
        # g_mol = g_mol.view(-1)  # (256,)

        # g_seq = dgl.add_self_loop(g_seq)
        # if edge_weight is None:
        #     edge_weight = g_seq.edata['weight']

        h_seq = self.conv1_seq(g_seq, feat_seq, edge_weight=g_seq.edata['weight'])
        h_seq = F.relu(h_seq)
        h_seq = self.conv2_seq(g_seq, h_seq, edge_weight=g_seq.edata['weight'])
        g_seq.ndata['h'] = h_seq
        hg_seq = dgl.mean_nodes(g_seq, 'h')

        if smiles.dim() == 1:
            smiles = smiles.unsqueeze(0)
        if kmer.dim() == 1:
            kmer = kmer.unsqueeze(0)  # 扩展维度至[batch_size, 1]
        features = torch.cat((hg_seq, g_mol, kmer), dim=1)
        features = features.to(torch.float32)
        out = self.fc1(features)
        out = F.relu(out)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        # out = F.relu(out)
        # out = self.fc5(out)

        return out, att_weights
