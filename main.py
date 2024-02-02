import torch
from utils.dgmimodel import DMGI
from utils.datahandle import DataIO
import argparse
import time
from indentify_cancer_gene import deepod_dsvdd
from embedding import train_embedding, get_embedding
import random
import numpy as np
from torch_geometric.data import DataLoader
# 设置随机种子
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()
# DMGI参数设置
# 添加一个参数，用于接收字典的键值对
parser.add_argument('--network_paths',
                    default=['PPI=./processed_dataset/six_net/PPI_for_pyG.txt',
                             'Pathway=./processed_dataset/six_net/Pathway_for_pyG.txt',
                             'Kinase=./processed_dataset/six_net/Kinase_for_pyG.txt',
                             'Metabolic=./processed_dataset/six_net/Metabolic_for_pyG.txt',
                             'Regulatory=./processed_dataset/six_net/Regulatory_for_pyG.txt',
                             'Complexes=./processed_dataset/six_net/Complexes_for_pyG.txt'],
                    nargs='+',
                    metavar='KEY=VALUE',
                    help='dictionary of network paths')
parser.add_argument('--feature_path',
                    default='./processed_dataset/six_net/Feature_for_pyG.csv',
                    help='Dictionary of feature path')
parser.add_argument('--mapping_dict',
                    default='./processed_dataset/six_net/mapping_dict_EIDtoIndex.pickle',
                    help='Dictionary of mapping dict path')
parser.add_argument("--kcg", default='./processed_dataset/six_net/kcg_intersec.csv', help='Know cancer gene')
parser.add_argument("--Maybe_cancergene", default="./processed_dataset/six_net/Merge_maybe_cancergene.txt")
parser.add_argument('--embedding_file_path', default="./saved_embedding/six_net", help='the file path of embedding')
parser.add_argument('--dimension', type=int, default=256, help='the dimension of embedding')
parser.add_argument('--lr_dmgi', type=float, default=0.001, help='learning rate of train')
parser.add_argument('--wd_dmgi', type=float, default=0.01, help='weight decay of train')

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
print('-----' + now + '-----')
print('-----0. Loading Data-----')
# 解析命令行参数
args = parser.parse_args()

# 创建一个空字典，用于存储传递的键值对
network_paths = {}
# 解析传递的键值对，并将其添加到字典中
if args.network_paths:
    for item in args.network_paths:
        key, value = item.split('=')
        network_paths[key] = value

dataIO = DataIO()
data, net_types = dataIO.load_network(
    Features_path=args.feature_path, need_features=True, **network_paths)  # 调用load_data方法
EID_to_index = dataIO.load_mapping_dict(args.mapping_dict)
kcg_intersec = dataIO.load_know_cancer_gene(args.kcg)
Maybe_cancergene_list = dataIO.load_maybe_cancer_gene(args.Maybe_cancergene)

node_types, edge_types = data.metadata()
edge_types_num = len(edge_types)
# 
# DMGI模型设置，设置训练参数
model_DMGI = DMGI(data['gene'].num_nodes, data['gene'].x.size(-1),
                  out_channels=args.dimension,
                  num_relations=edge_types_num)  # num_relations 这里要node_types, edge_types = data.metadata(),len(len(edge_types))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model_DMGI = data.to(device), model_DMGI.to(device) #数据、模型都需要放到GPU
print(model_DMGI) #输出网络结构
optimizer_DMGI = torch.optim.RMSprop(model_DMGI.parameters(), lr=args.lr_dmgi, weight_decay=args.wd_dmgi)


if __name__ == '__main__':
    print(f"Networks: {list(network_paths.keys())}")

    ###训练向量:
    print("-----1. Training Embedding-----")
    batch_size = 16
    #data_loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    data_loader = data
    train_embedding(model_DMGI, optimizer_DMGI, data_loader, net_types)
    print("-----2. Getting Embedding-----")
    embedding = get_embedding(model_DMGI)  # get embedding
    encoded_embedding = embedding
    #dataIO.save_embedding(embedding) #保存嵌入向量
    print("Saved!")

    ###使用DEEPOD中的DEEP SVDD：
    cancer_genes_proba = deepod_dsvdd(kcg_intersec, encoded_embedding, EID_to_index, Maybe_cancergene_list)
    print("Already Done!")


