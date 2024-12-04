import torch
from torch.utils.data import DataLoader, Dataset


# 批次Dataset类
class CustomDataset(Dataset):
    def __init__(self, graphs_mol, graphs_seq, smiles, labels):
        # self.rna_data = rna_data
        self.graphs_mol = graphs_mol
        self.graphs_seq = graphs_seq
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs_mol[idx], self.graphs_seq[idx], self.smiles[idx], self.labels[idx]


def custom_collate_fn(batch):
    graphs_mol, graphs_seq, smiles, labels = zip(*batch)
    smiles = torch.stack(smiles)
    labels = torch.stack(labels)
    return graphs_mol, graphs_seq, smiles, labels
