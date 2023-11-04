import torch
from torch_geometric.data import HeteroData
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import pickle
import os

class DataIO():
    def __init__(self):
        pass

    def load_network(self, Features_path, need_features=True, **kwargs):
        
        data = HeteroData()
        
        Features = pd.read_csv(Features_path)  # 特征
        Features = Features.sort_values(by='EnterzID')
        x = np.array(Features.iloc[:, 1:])  #从第一列开始
        x = torch.tensor(x, dtype=torch.float)
        if need_features == False: #已经集成了不需要特征时的处理方法
            x.fill_(1)
        #print(x[:5]) #预览数据
        data['gene'].x = x
        
        net_types = []
        for net_name, net_path in kwargs.items():
            net_adj = pd.read_csv(net_path, header=None, sep='\t')
            head = list(net_adj[0])
            tail = list(net_adj[1])
            edge_index = torch.tensor([head, tail], dtype=torch.long)
            data['gene', net_name, 'gene'].edge_index = edge_index
            net_types.append(net_name)

        return data, net_types
    
    @torch.no_grad()
    def save_embedding(self, embedding, path="./saved_embedding"):
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        tensor_data = embedding.data  # Extract tensor data from the Parameter object, model.Z的类型是一个Parameter
        torch.save(tensor_data, f"{path}/embedding_{now}.pt")

        return f"{path}/embedding_{now}.pt"
    
    def load_embedding(self, path):
        try:
            embedding = torch.load(path)
            return embedding
        except FileNotFoundError:
            print("无法找到嵌入数据文件。请检查路径是否正确。")
            return None

    def load_mapping_dict(self, mapping_dict_path):
        file = open(mapping_dict_path, "rb")
        EID_to_index_dict = pickle.load(file)
        
        return EID_to_index_dict
    
    def load_know_cancer_gene(self, path):
        kcg_intersec = pd.read_csv(path, header=None)
        return kcg_intersec
        
    def load_maybe_cancer_gene(self, path):
        Maybe_cancergene = pd.read_csv(path, header=None)
        Maybe_cancergene_list = Maybe_cancergene[0].to_list()

        return Maybe_cancergene_list

