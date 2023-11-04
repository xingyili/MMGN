import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def train_embedding(model, optimizer, data_loader, net_types):
    def train(batch):
        model.train()
        optimizer.zero_grad()
        x = batch['gene'].x
        edge_indices = [batch['gene', net_name, 'gene'].edge_index for net_name in net_types]
        pos_hs, neg_hs, summaries = model(x, edge_indices)
        loss, loss1 = model.loss(pos_hs, neg_hs, summaries)
        loss.backward()
        optimizer.step()
        return float(loss), float(loss1)

    patience = 20
    best = 1e9
    cnt_wait = 0

    for epoch in range(1, 10001):
        for batch in data_loader:
            loss, loss1 = train(batch)
        if epoch % 50 == 0:
            # val_acc, test_acc = test()
            print(f'Epoch: {epoch:03d}, Loss_ALL: {loss:.4f}')

        if loss < best:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break
            
@torch.no_grad()
def get_embedding(model):
    return model.Z.detach().clone().cpu()



    

