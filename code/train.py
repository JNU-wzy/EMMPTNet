import os
import time
import logging
import dgl
import numpy as np
import pandas as pd
from dgl.data import Subset
from torch.utils.data import DataLoader
from Net import *
import torch
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from evaluation import *
from sklearn.model_selection import KFold
from MOL_features import *
from Batch_Dataset import *
import pandas as pd
import pickle


def train(train_graphs_mol, train_graphs_seq, train_smiles, train_labels, train_number, device, test_graphs_mol,
          test_graphs_seq, test_labels):
    save = "result/"
    model_filename = f"withH_model_selfatt_{train_number + 1}_xavier_fast-5.pth"
    if os.path.exists(os.path.join(save, model_filename)):
        print("model exists")
        return
    model = GCN_BiLSTM_selfattn().to(device)
    reset_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-5)
    best_RMSE_loss = 10
    patience = 50  
    counter = 0  
    loss_history = []

    train_dataset = CustomDataset(train_graphs_mol, train_graphs_seq, train_smiles, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for epoch in range(100):
        start_time = time.time()
        model.train()
        total_loss = 0
        for g_mol, g_seq, s, labels in train_loader:
            g_mol = [graph.to(device) for graph in g_mol]
            g_seq = [graph.to(device) for graph in g_seq]
            s = s.to(device)
            labels = labels.to(device).float()

            features_seq = [graph.ndata['attr'].to(device).float() for graph in g_seq]
            # features_mol = [g.ndata['feat'].to(device).float() for g in g_mol]

            preds = []
            for mol, seq, feat_seq, sm in zip(g_mol, g_seq, features_seq, s):
                pred = model(mol, seq, feat_seq, sm)
                preds.append(pred)

            preds = [x[0] for x in preds]
            preds = torch.stack(preds).squeeze()

            if torch.isnan(preds).any():
                print(f"Training stopped: NaN detected in outputs at epoch {epoch}")
                return

            loss = rmse(labels, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch}: Train Loss: {total_loss / len(train_loader)}, Duration: {epoch_duration:.2f} seconds")
        loss_history.append(total_loss / len(train_loader))
       
        if total_loss / len(train_loader) < best_RMSE_loss:
            best_RMSE_loss = total_loss / len(train_loader)
            counter = 0  
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(save, model_filename))
            print(f"Saved model at epoch {epoch}")
        else:
            counter += 1

        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}, no improvement in training loss for {patience} epochs.")
            break

    print("Training complete.")


def test(test_data_seq, test_data_mol, test_smiles, test_labels, model_path, device):
   
    test_labels = test_labels.clone().detach().to(device).float()

   
    model = GCN_BiLSTM_selfattn()
    checkpoint = torch.load(model_path)
    # for key in checkpoint['state_dict']:
    #     print(f"{key}: {checkpoint['state_dict'][key].shape}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        test_preds = [
            model(g_mol.to(device), g_seq.to(device), g_seq.ndata['attr'].to(device),  s.to(device))[0].squeeze()
            for (g_seq, g_mol, s) in zip(test_data_seq, test_data_mol, test_smiles)]
        test_preds = torch.stack(test_preds).squeeze().cpu().numpy()
        # val_labels = val_labels.cpu().numpy()
        if isinstance(test_labels, np.ndarray):
            test_labels = test_labels
        else:
            # 如果val_labels是一个PyTorch张量，则转换为NumPy数组
            test_labels = test_labels.cpu().numpy()

    RMSE = root_mean_squared_error(test_labels, test_preds)
    PCC = pearson_correlation_coefficient(test_labels, test_preds)
    SPCC = spearman_rank_correlation_coefficient(test_labels, test_preds)
    MAE = mean_absolute_error(test_labels, test_preds)
    print(f"TEST---RMSE: {RMSE}, PCC: {PCC}, SPCC: {SPCC}, MAE: {MAE}")

    return RMSE, PCC, SPCC, MAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

graphs_seq = dgl.load_graphs(r"D:\Project\bioinformatics_project\binding "
                             r"affinity\code\smiles_kmer_GCNseq_GCNMOL\dataset\fasttext_k5_d128_rna.bin")
graphs_seq = graphs_seq[0]
labels = torch.Tensor(np.load("dataset/labels.npy")).to(device)
smiles = pd.read_excel("dataset/smiles_3ker.xlsx")

file_path = 'dataset/mol_withH_17.pkl'  # **


with open(file_path, 'rb') as file:
    graphs_mol = pickle.load(file)
with open('train_test_splits.pkl', 'rb') as f:
    train_test_splits = pickle.load(f)

#kfold = KFold(n_splits=10, shuffle=True, random_state=42)
train_number = 0
total_rmse = []
total_pcc = []
total_spcc = []
total_mae = []
for train_idx, test_idx in train_test_splits:
    print("Train times: ", train_number + 1)
    # if (train_number + 1) != 4:
    #     print("!=4")
    #     train_number += 1
    #     continue
    train_graphs_mol = [graphs_mol[i] for i in train_idx]
    train_graphs_seq = [graphs_seq[i] for i in train_idx]
    train_smiles = [torch.tensor(smiles.iloc[i].values) for i in train_idx]
    test_graphs_mol = [graphs_mol[i] for i in test_idx]
    test_graphs_seq = [graphs_seq[i] for i in test_idx]
    test_smiles = [torch.tensor(smiles.iloc[i].values) for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train(train_graphs_mol, train_graphs_seq, train_smiles, train_labels, train_number, device, test_graphs_mol,
          test_graphs_seq, test_labels)
    model_path = f"result/withH_model_selfatt_{train_number + 1}_xavier_fast-5.pth"
    RMSE, PCC, SPCC, MAE = test(test_graphs_seq, test_graphs_mol, test_smiles, test_labels, model_path, device)
    train_number += 1

    total_rmse.append(RMSE)
    total_pcc.append(PCC)
    total_spcc.append(SPCC)
    total_mae.append(MAE)
    print("Train End")

average_rmse = np.mean(total_rmse)
average_pcc = np.mean(total_pcc)
average_spcc = np.mean(total_spcc)
average_mae = np.mean(total_mae)
print(f'Average RMSE: {average_rmse}, std： {np.std(total_rmse)}')
print(f'Average PCC: {average_pcc}, std： {np.std(total_pcc)}')
print(f'Average SPCC: {average_spcc}, std： {np.std(total_spcc)}')
print(f'Average MAE: {average_mae}, std： {np.std(total_mae)}')
