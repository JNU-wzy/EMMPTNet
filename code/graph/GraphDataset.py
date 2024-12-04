import os
import pickle
from collections import Counter
import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
import numpy as np
from dgl.data.utils import save_info, load_info
from dgl.nn.pytorch import EdgeWeightNorm
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from utils.config import *
import pandas as pd
import glob
import os

params = config()


class TrainDataset(DGLDataset):
    """
        url : str
            The url to download the original dataset.
        raw_dir : str
            Specifies the directory where the downloaded data is stored or where the downloaded data is stored. Default: ~/.dgl/
        save_dir : str
            The directory where the finished dataset will be saved. Default: the value specified by raw_dir
        force_reload : bool
            If or not to re-import the dataset. Default: False
        verbose : bool
            Whether to print progress information.
        """

    def __init__(self, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=False):
        super(TrainDataset, self).__init__(name='rna',
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

        print('***Executing init function***')
        print('Dataset initialization is completed!\n')

    def process(self):
        # Processing of raw data into plots, labels
        print('***Executing process function***')
        self.kmers = params.k

        # Open files and load data
        print('Loading the raw data...')
        f_g = glob.glob(self.raw_dir)
        # 初始化空列表来存储数据和标签
        data = []
        rawLab = []

        # 循环遍历所有Excel文件
        for file_path in f_g:
            # if self.raw_dir is not None:
                #file_path = self.raw_dir
                # 读取当前文件
                df = pd.read_excel(file_path)

                # 提取Target_RNA_sequence列和pKd列
                data.extend(df['Target_RNA_sequence'].apply(lambda x: x.replace(" ", "")).tolist())

                # 处理可能的pKd格式问题，如将字符串转换为浮点数
                pKd_values = df['pKd'].apply(lambda x: float(str(x).replace(" ", "").split('\n')[0])).tolist()
                rawLab.extend(pKd_values)
        rawLab = np.array(rawLab)
        arr_min = np.min(rawLab)
        arr_max = np.max(rawLab)
        rawLab = (rawLab - arr_min) / (arr_max - arr_min)

        # Get labels and k-mer sentences 将序列转换为k-mer，用于之后的图构建
        k_RNA = [[i[j:j + self.kmers] for j in range(len(i) - self.kmers + 1)] for i in data]

        # Get the mapping variables for kmers and kmers_id
        print('Getting the mapping variables for kmers and kmers id...')
        self.kmers2id, self.id2kmers = {"<EOS>": 0}, ["<EOS>"]
        kmersCnt = 1
        for rna in tqdm(k_RNA):
            for kmers in rna:
                if kmers not in self.kmers2id:
                    self.kmers2id[kmers] = kmersCnt
                    self.id2kmers.append(kmers)
                    kmersCnt += 1
        self.kmersNum = kmersCnt

        # Get the ids of RNAsequence and label
        self.k_RNA = k_RNA
        self.labels = t.tensor(rawLab)
        self.idSeq = np.array([[self.kmers2id[i] for i in s] for s in self.k_RNA], dtype=object)  # 由K-mer构成的序列，转换成数字

        self.vectorize()

        # Construct and save the graph
        self.graphs = []
        for eachseq in self.idSeq:
            newidSeq = []  # 用于存储转换后的序列ID
            old2new = {}  # 记录序列中每个唯一k-mer的新ID （为这一个序列重新编号）
            count = 0
            for eachid in eachseq:
                if eachid not in old2new:
                    old2new[eachid] = count
                    count += 1
                newidSeq.append(old2new[eachid])
                """遍历当前序列中的每个k - merID（eachid），如果这个ID还没有在old2new映射中，就为它分配一个新的ID，
                并将这个新ID添加到newidSeq中。这一步确保了每个k - mer在图中都有一个唯一的节点表示"""

            # print(newidSeq)

            counter_uv = Counter(list(zip(newidSeq[:-1], newidSeq[
                                                         1:])))  # 计算边的权重 使用zip(newidSeq[:-1], newidSeq[1:])生成一对相邻元素的元组，这代表图中的边，Counter来统计序列中相邻元素（k-mer ID）对出现的频次
            """
            newidSeq = [0, 1, 2, 1, 0, 3, 0, 1]
            zip(...) = [(0, 1), (1, 2), (2, 1), (1, 0), (0, 3), (3, 0), (0, 1)]
            counter_uv = Counter([(0, 1), (1, 2), (2, 1), (1, 0), (0, 3), (3, 0), (0, 1)])
            --> Counter({(0, 1): 2, (1, 2): 1, (2, 1): 1, (1, 0): 1, (0, 3): 1, (3, 0): 1})
            counter_uv.keys = dict_keys([(0, 1), (1, 2), (2, 1), (1, 0), (0, 3), (3, 0)])
            """
            graph = dgl.graph(list(counter_uv.keys()))  # 自动识别边的索引有几个，就有几个节点，有几个边

            weight = t.FloatTensor(list(counter_uv.values()))  # 归一化
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph, weight)
            graph.edata['weight'] = norm_weight
            # print(self.vector)
            # print(list(old2new.keys()))
            node_features = self.vector['embedding'][list(old2new.keys())]
            graph.ndata['attr'] = t.tensor(node_features)
            self.graphs.append(graph)

    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graphs[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graphs)

    def save(self):
        # Save the processed data to `self.save_path`
        print('***Executing save function***')
        save_graphs(self.save_dir + ".bin", self.graphs, {'labels': self.labels})
        # Save additional information in the Python dictionary
        # info_path = self.save_dir + "_info.pkl"
        # info = {'kmers': self.kmers, 'kmers2id': self.kmers2id, 'id2kmers': self.id2kmers}
        # save_info(info_path, info)

    def load(self):
        # Import processed data from `self.save_path`
        print('***Executing load function***')
        self.graphs, label_dict = load_graphs(self.save_dir + ".bin")
        self.labels = label_dict['labels']
        # self.labels = label_dict['labels']
        # info_path = self.save_dir + "_info.pkl"
        # info = load_info(info_path)
        # self.kmers, self.kmers2id, self.id2kmers, self.lab2id, self.id2lab = info['kmers'], info['kmers2id'], info[
        #     'id2kmers'], info['lab2id'], info['id2lab']

    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        print('***Executing has_cache function***')
        graph_path = self.save_dir + ".bin"
        info_path = self.save_dir + "_info.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def vectorize(self, method="kmers", feaSize=params.d, window=5, sg=1,
                  workers=8, loadCache=True):
        self.vector = {}
        # print(f"Class name is set to: {self.class_name}")
        print('\n***Executing vectorize function***')
        if os.path.exists(f'Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl') and loadCache:
            with open(f'Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl', 'rb') as f:
                if method == 'kmers':
                    tmp = pickle.load(f)
                    self.vector['encoder'], self.kmersFea = tmp['encoder'], tmp['kmersFea']
                else:
                    self.vector['embedding'] = pickle.load(f)
            print(f'Load cache from Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl!')
            return
        if method == 'word2vec':
            doc = [i + ['<EOS>'] for i in self.k_RNA]
            model = Word2Vec(doc, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
            word2vec = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            for i in range(self.kmersNum):
                word2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = word2vec
        elif method == 'kmers':
            enc = OneHotEncoder(categories='auto')
            enc.fit([[i] for i in self.kmers2id.values()])
            feaSize = len(self.kmers2id)
            kmers = np.zeros((len(self.labels), feaSize))
            bs = 50000
            print('Getting the kmers vector...')
            for i, t in enumerate(self.idSeq):
                for j in range((len(t) + bs - 1) // bs):
                    kmers[i] += enc.transform(np.array(t[j * bs:(j + 1) * bs]).reshape(-1, 1)).toarray().sum(
                        axis=0)
            kmers = kmers[:, 1:]
            feaSize -= 1
            # Normalized
            kmers = (kmers - kmers.mean(axis=0)) / kmers.std(axis=0)
            self.vector['encoder'] = enc
            self.kmersFea = kmers

        # Save k-mer vectors
        with open(f'Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl', 'wb') as f:
            if method == 'kmers':
                pickle.dump({'encoder': self.vector['encoder'], 'kmersFea': self.kmersFea}, f, protocol=4)
            else:
                pickle.dump(self.vector['embedding'], f, protocol=4)


class TestDataset(DGLDataset):
    """
        url : str
            The url to download the original dataset.
        raw_dir : str
            Specifies the directory where the downloaded data is stored or where the downloaded data is stored. Default: ~/.dgl/
        save_dir : str
            The directory where the finished dataset will be saved. Default: the value specified by raw_dir
        force_reload : bool
            If or not to re-import the dataset. Default: False
        verbose : bool
            Whether to print progress information.
        """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(TestDataset, self).__init__(name='lncrna',
                                          url=url,
                                          raw_dir=raw_dir,
                                          save_dir=save_dir,
                                          force_reload=force_reload,
                                          verbose=verbose
                                          )
        print('***Executing init function***')
        print('Dataset initialization is completed!\n')

    def process(self):
        # Processing of raw data into plots, labels
        print('***Executing process function***')
        self.kmers = params.k
        # Open files and load data
        print('Loading the raw data...')
        f_g = glob.glob(self.raw_dir)
        # 初始化空列表来存储数据和标签
        data = []
        rawLab = []

        # 循环遍历所有Excel文件
        for file_path in f_g:
            # 读取当前文件
            df = pd.read_excel(file_path)

            # 提取Target_RNA_sequence列和pKd列
            data.extend(df['Target_RNA_sequence'].apply(lambda x: x.replace(" ", "")).tolist())

            # 处理可能的pKd格式问题，如将字符串转换为浮点数
            pKd_values = df['pKd'].apply(lambda x: float(str(x).replace(" ", "").split('\n')[0])).tolist()
            rawLab.extend(pKd_values)
        rawLab = np.array(rawLab)
        arr_min = np.min(rawLab)
        arr_max = np.max(rawLab)
        rawLab = (rawLab - arr_min) / (arr_max - arr_min)

        # Get labels and k-mer sentences 将序列转换为k-mer，用于之后的图构建
        k_RNA = [[i[j:j + self.kmers] for j in range(len(i) - self.kmers + 1)] for i in data]

        # Get the mapping variables for kmers and kmers_id
        print('Getting the mapping variables for kmers and kmers id...')
        self.kmers2id, self.id2kmers = {"<EOS>": 0}, ["<EOS>"]
        kmersCnt = 1
        for rna in tqdm(k_RNA):
            for kmers in rna:
                if kmers not in self.kmers2id:
                    self.kmers2id[kmers] = kmersCnt
                    self.id2kmers.append(kmers)
                    kmersCnt += 1
        self.kmersNum = kmersCnt

        # Get the ids of RNAsequence and label
        self.k_RNA = k_RNA
        self.labels = t.tensor(rawLab)
        self.idSeq = np.array([[self.kmers2id[i] for i in s] for s in self.k_RNA], dtype=object)  # 由K-mer构成的序列，转换成数字

        self.vectorize()

        # Construct and save the graph
        self.graphs = []
        for eachseq in self.idSeq:
            newidSeq = []  # 用于存储转换后的序列ID
            old2new = {}  # 记录序列中每个唯一k-mer的新ID （为这一个序列重新编号）
            count = 0
            for eachid in eachseq:
                if eachid not in old2new:
                    old2new[eachid] = count
                    count += 1
                newidSeq.append(old2new[eachid])
                """遍历当前序列中的每个k - merID（eachid），如果这个ID还没有在old2new映射中，就为它分配一个新的ID，
                并将这个新ID添加到newidSeq中。这一步确保了每个k - mer在图中都有一个唯一的节点表示"""

            # print(newidSeq)

            counter_uv = Counter(list(zip(newidSeq[:-1], newidSeq[1:])))  # 计算边的权重
            graph = dgl.graph(list(counter_uv.keys()))

            weight = t.FloatTensor(list(counter_uv.values()))  # 归一化
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph, weight)
            graph.edata['weight'] = norm_weight
            # print(self.vector)
            # print(list(old2new.keys()))
            node_features = self.vector['embedding'][list(old2new.keys())]
            graph.ndata['attr'] = t.tensor(node_features)
            self.graphs.append(graph)

    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graphs[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graphs)

    def save(self):
        # Save the processed data to `self.save_path`
        print('***Executing save function***')
        save_graphs(self.save_dir + ".bin", self.graphs, {'labels': self.labels})
        # Save additional information in the Python dictionary
        # info_path = self.save_dir + "_info.pkl"
        # info = {'kmers': self.kmers, 'kmers2id': self.kmers2id, 'id2kmers': self.id2kmers}
        # save_info(info_path, info)

    def load(self):
        # Import processed data from `self.save_path`
        print('***Executing load function***')
        self.graphs, label_dict = load_graphs(self.save_dir + ".bin")
        self.labels = label_dict['labels']
        # self.labels = label_dict['labels']
        # info_path = self.save_dir + "_info.pkl"
        # info = load_info(info_path)
        # self.kmers, self.kmers2id, self.id2kmers, self.lab2id, self.id2lab = info['kmers'], info['kmers2id'], info[
        #     'id2kmers'], info['lab2id'], info['id2lab']

    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        print('***Executing has_cache function***')
        graph_path = self.save_dir + ".bin"
        info_path = self.save_dir + "_info.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def vectorize(self, method="word2vec", feaSize=params.d, window=5, sg=1,
                  workers=8, loadCache=True):
        self.vector = {}
        print('\n***Executing vectorize function***')
        if os.path.exists(f'Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl') and loadCache:
            with open(f'Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl', 'rb') as f:
                if method == 'kmers':
                    tmp = pickle.load(f)
                    self.vector['encoder'], self.kmersFea = tmp['encoder'], tmp['kmersFea']
                else:
                    self.vector['embedding'] = pickle.load(f)
            print(f'Load cache from Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl!')
            return
        if method == 'word2vec':
            doc = [i + ['<EOS>'] for i in self.k_RNA]
            model = Word2Vec(doc, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
            word2vec = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            for i in range(self.kmersNum):
                word2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = word2vec
        elif method == 'kmers':
            enc = OneHotEncoder(categories='auto')
            enc.fit([[i] for i in self.kmers2id.values()])
            feaSize = len(self.kmers2id)
            kmers = np.zeros((len(self.labels), feaSize))
            bs = 50000
            print('Getting the kmers vector...')
            for i, t in enumerate(self.idSeq):
                for j in range((len(t) + bs - 1) // bs):
                    kmers[i] += enc.transform(np.array(t[j * bs:(j + 1) * bs]).reshape(-1, 1)).toarray().sum(
                        axis=0)
            kmers = kmers[:, 1:]
            feaSize -= 1
            # Normalized
            kmers = (kmers - kmers.mean(axis=0)) / kmers.std(axis=0)
            self.vector['encoder'] = enc
            self.kmersFea = kmers

        # Save k-mer vectors
        with open(f'Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl', 'wb') as f:
            if method == 'kmers':
                pickle.dump({'encoder': self.vector['encoder'], 'kmersFea': self.kmersFea}, f, protocol=4)
            else:
                pickle.dump(self.vector['embedding'], f, protocol=4)



params = config()
dataset = TrainDataset(raw_dir=r"D:\Project\Pytorch_project\binding affinity\R-SIM\train\Excel\cleaned\Viral_RNA.xlsx", save_dir=rf'D:\Project\Pytorch_project\binding affinity\code\graph\dglgraph\onehot_k{params.k}_d{params.d}_smiles')