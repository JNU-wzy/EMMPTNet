import re

import torch
from rdkit import Chem
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import glob
import pickle


def extract_number(filename):
    # 正则表达式查找文件名中的数字
    match = re.search(r'molecule_(\d+)', filename)
    if match:
        number = int(match.group(1))
        # print(f"Extracted number {number} from {filename}")  # 调试输出
        return number
    else:
        print(f"No number found in {filename}")  # 调试输出
        return 0


# 原子性质字典
properties = {
    'C': {'atomic_radius': 0.77, 'mass': 12.01, 'electronegativity': 2.55, 'valence': 4, 'period': 2, 'group': 14,
          'ionic_radius': 0.16, 'ionization_energy': 11.26, 'electron_affinity': 1.26},
    'O': {'atomic_radius': 0.73, 'mass': 16.00, 'electronegativity': 3.44, 'valence': 2, 'period': 2, 'group': 16,
          'ionic_radius': 1.40, 'ionization_energy': 13.62, 'electron_affinity': 1.46},
    'H': {'atomic_radius': 0.37, 'mass': 1.008, 'electronegativity': 2.20, 'valence': 1, 'period': 1, 'group': 1,
          'ionic_radius': 0.08, 'ionization_energy': 13.60, 'electron_affinity': 0.75},
    'N': {'atomic_radius': 0.75, 'mass': 14.01, 'electronegativity': 3.04, 'valence': 3, 'period': 2, 'group': 15,
          'ionic_radius': 0.13, 'ionization_energy': 14.53, 'electron_affinity': -0.07},
    'S': {'atomic_radius': 1.02, 'mass': 32.06, 'electronegativity': 2.58, 'valence': 2, 'period': 3, 'group': 16,
          'ionic_radius': 1.84, 'ionization_energy': 10.36, 'electron_affinity': 2.07},
    'Cl': {'atomic_radius': 0.99, 'mass': 35.45, 'electronegativity': 3.16, 'valence': 1, 'period': 3, 'group': 17,
           'ionic_radius': 1.81, 'ionization_energy': 12.97, 'electron_affinity': 3.61},
    'P': {'atomic_radius': 1.06, 'mass': 30.97, 'electronegativity': 2.19, 'valence': 3, 'period': 3, 'group': 15,
          'ionic_radius': 0.38, 'ionization_energy': 10.49, 'electron_affinity': 0.74},
    'Co': {'atomic_radius': 1.16, 'mass': 58.93, 'electronegativity': 1.88, 'valence': 2, 'period': 4, 'group': 9,
           'ionic_radius': 0.745, 'ionization_energy': 7.86, 'electron_affinity': 0.66},
    'F': {'atomic_radius': 0.72, 'mass': 18.998, 'electronegativity': 3.98, 'valence': 1, 'period': 2, 'group': 17,
          'ionic_radius': 1.33, 'ionization_energy': 17.42, 'electron_affinity': 3.40},
    'I': {'atomic_radius': 1.33, 'mass': 126.90, 'electronegativity': 2.66, 'valence': 1, 'period': 5, 'group': 17,
          'ionic_radius': 2.20, 'ionization_energy': 10.45, 'electron_affinity': 3.06},
    'Br': {'atomic_radius': 1.14, 'mass': 79.904, 'electronegativity': 2.96, 'valence': 1, 'period': 4, 'group': 17,
           'ionic_radius': 1.96, 'ionization_energy': 11.81, 'electron_affinity': 3.36},
    'Ru': {'atomic_radius': 1.34, 'mass': 101.07, 'electronegativity': 2.2, 'valence': 3, 'period': 5, 'group': 8,
           'ionic_radius': 0.68, 'ionization_energy': 7.36, 'electron_affinity': 1.05}
}


# 提取单个MOL文件中的特征
def extract_features(mol_file):
    mol = Chem.MolFromMolFile(mol_file)

    if mol is None:
        raise ValueError(f"MOL file {mol_file} could not be read")

    # 添加显式氢原子
    mol = Chem.AddHs(mol)

    # 获取分子中的原子数量
    num_atoms = mol.GetNumAtoms()

    # 初始化特征矩阵
    num_features = 14  # 根据特性数量调整
    features = np.zeros((num_atoms, num_features))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()

        # 获取基本原子信息
        atomic_num = atom.GetAtomicNum()  # 原子序数
        is_in_ring = atom.IsInRing()  # 是否在环中
        formal_charge = atom.GetFormalCharge()  # 形式电荷
        num_free_electrons = atom.GetNumImplicitHs()  # 自由电子数

        # 将基本信息填入特征矩阵
        features[idx, 0] = atomic_num
        features[idx, 1] = 1  # 使用1表示原子类型存在
        features[idx, 2] = int(is_in_ring)
        features[idx, 3] = formal_charge
        features[idx, 4] = num_free_electrons

        # 获取其他原子特性
        if symbol in properties:
            atom_props = properties[symbol]
            features[idx, 5] = atom_props['atomic_radius']
            features[idx, 6] = atom_props['mass']
            features[idx, 7] = atom_props['electronegativity']
            features[idx, 8] = atom_props['valence']
            features[idx, 9] = atom_props['period']
            features[idx, 10] = atom_props['group']
            features[idx, 11] = atom_props['ionic_radius']
            features[idx, 12] = atom_props['ionization_energy']
            features[idx, 13] = atom_props['electron_affinity']
        else:
            # 如果原子不在字典中，用-1表示
            features[idx, 5:] = -1

    return features


# 自定义Dataset类
class MolDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = self._process_files()

    def _process_files(self):
        data = []
        for mol_file in self.folder_path:
            try:
                features = extract_features(mol_file)
                data.append(features)
            except ValueError as e:
                print(e)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        return torch.tensor(features, dtype=torch.float32)


# 批量处理文件夹中的所有MOL文件
def process_folder(folder_path):
    data = {}

    for mol_file in file_path:
        try:
            features = extract_features(mol_file)
            data[mol_file] = features
        except ValueError as e:
            print(e)

    return data

# f_g = glob.glob(folder_path + "/molecule_*.mol")
# # 对文件路径列表进行排序，根据文件名中的数字
# file_paths = sorted(f_g, key=extract_number)
# dataset = MolDataset(file_paths)
#
#
# with open('dataset/mol_withH.pkl', 'wb') as f:
#     pickle.dump(dataset, f)
